"""Gibraltar three-panel visual AI dashboard using three camera sources.

Each configured RTSP or USB camera feeds its respective OCR/object-detection
worker and panel. Sources are managed from the GUI and persisted in config.json.
"""
####.\.venv\Scripts\python.exe Projects/main.py

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent
CONFIG_PATH = PROJECT_DIR / "config.json"
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from camera import LocalCamera, RTSPCamera  # noqa: E402
from communication.inspection_sender import InspectionSender  # noqa: E402
from communication.socket_client import SocketClient  # noqa: E402


def load_project_config() -> dict[str, Any]:
    """Load Gibraltar settings without failing GUI startup."""
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            loaded = json.load(config_file)
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def load_image_repository_path() -> Path:
    """Load the external snapshot path from the project configuration."""
    default_path = Path.home() / "Pictures" / "Gibraltar" / "image_repository"
    configured = load_project_config().get("image_repository_path")
    if configured:
        expanded = os.path.expandvars(os.path.expanduser(str(configured)))
        return Path(expanded).resolve()
    return default_path.resolve()


def save_image_repository_path(repository_path: Path) -> None:
    """Persist the user's selected external snapshot path."""
    config = load_project_config()
    config["image_repository_path"] = str(repository_path.resolve())
    temporary_path = CONFIG_PATH.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=4)
        config_file.write("\n")
    temporary_path.replace(CONFIG_PATH)


def load_camera_sources() -> list[dict[str, str]]:
    """Load three camera sources, including legacy ``camera_urls`` values."""
    config = load_project_config()
    configured = config.get("camera_sources")
    sources: list[dict[str, str]] = []
    if isinstance(configured, list):
        for item in configured[:3]:
            if not isinstance(item, dict):
                sources.append({"type": "rtsp", "value": ""})
                continue
            source_type = str(item.get("type", "rtsp")).strip().lower()
            if source_type not in {"rtsp", "usb"}:
                source_type = "rtsp"
            sources.append(
                {"type": source_type, "value": str(item.get("value", "")).strip()}
            )
    else:
        legacy_urls = config.get("camera_urls", [])
        if isinstance(legacy_urls, list):
            sources = [
                {"type": "rtsp", "value": str(value).strip()}
                for value in legacy_urls[:3]
            ]
    return sources + [
        {"type": "rtsp", "value": ""}
        for _ in range(3 - len(sources))
    ]


def save_camera_sources(camera_sources: list[dict[str, str]]) -> None:
    """Persist the three independent camera sources."""
    config = load_project_config()
    config["camera_sources"] = camera_sources
    config.pop("camera_urls", None)
    temporary_path = CONFIG_PATH.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=4)
        config_file.write("\n")
    temporary_path.replace(CONFIG_PATH)


def latest_put(target_queue: Any, value: Any) -> None:
    """Put a value without allowing old frames to build up."""
    try:
        target_queue.put_nowait(value)
        return
    except queue.Full:
        pass
    try:
        target_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        target_queue.put_nowait(value)
    except queue.Full:
        pass


def encode_frame(frame: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    if not success:
        raise RuntimeError("Could not encode camera frame")
    return encoded.tobytes()


def decode_frame(data: bytes) -> np.ndarray:
    frame = cv2.imdecode(
        np.frombuffer(data, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if frame is None:
        raise RuntimeError("Could not decode camera frame")
    return frame


def extract_visual_feature(frame: np.ndarray) -> np.ndarray:
    """Create a compact colour/appearance feature for reference matching."""
    resized = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [12, 8, 8],
        [0, 180, 0, 256, 0, 256],
    )
    return cv2.normalize(histogram, histogram).flatten().astype(np.float32)


def visual_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Return histogram correlation normalized to the 0..1 range."""
    correlation = cv2.compareHist(
        first.astype(np.float32),
        second.astype(np.float32),
        cv2.HISTCMP_CORREL,
    )
    return max(0.0, min(1.0, (float(correlation) + 1.0) / 2.0))


def create_ocr_processor(
    device: str,
) -> Callable[[np.ndarray], tuple[np.ndarray, dict[str, Any]]]:
    from OCR.PaddleOCR import OCR, OCRConfig
    from OCR.PaddleOCR.drawing import draw_result

    reader = OCR(OCRConfig(device=device))

    def process(frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        result = reader.read(frame)
        output = draw_result(frame, result, copy=False)
        cv2.putText(
            output,
            f"OCR {result.inference_time_ms:.0f} ms",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return output, {"ocr": result.as_dict()}

    return process


def create_detection_processor(
    device: str,
) -> Callable[[np.ndarray], np.ndarray]:
    from models.detector import ObjectDetector

    detector = ObjectDetector(
        str(REPOSITORY_DIR / "models" / "weights" / "yolov8n.pt"),
        device=device,
    )

    def process(frame: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        detections, results = detector.detect(frame)
        output = detector.draw(frame, results)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        cv2.putText(
            output,
            f"Objects {len(detections)} | {elapsed_ms:.0f} ms",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    return process


def processing_worker(
    kind: str,
    input_queue: Any,
    output_queue: Any,
    stop_event: Any,
    device: str,
) -> None:
    """Run one AI pipeline in a child process."""
    # The GUI parent owns shutdown. Console signals sent to the whole Windows
    # process group must not interrupt a worker halfway through native Paddle
    # or PyTorch initialization.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    try:
        latest_put(output_queue, ("status", "Loading model..."))
        processor = (
            create_ocr_processor(device)
            if kind == "ocr"
            else create_detection_processor(device)
        )
        latest_put(output_queue, ("status", "Ready"))

        while not stop_event.is_set():
            try:
                encoded = input_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if encoded is None:
                break
            frame = decode_frame(encoded)
            processed = processor(frame)
            if isinstance(processed, tuple):
                output, metadata = processed
            else:
                output, metadata = processed, {}
            latest_put(
                output_queue,
                ("frame", (encode_frame(output), metadata)),
            )
    except Exception as error:
        if not stop_event.is_set():
            latest_put(
                output_queue,
                ("error", f"{type(error).__name__}: {error}"),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gibraltar three-camera RTSP visual AI dashboard",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--ocr-device", default="gpu:0")
    parser.add_argument("--object-device", default="cpu")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("width, height, and fps must be positive")
    return args


def run_gui(args: argparse.Namespace) -> int:
    from PySide6.QtCore import QObject, QTimer, Qt, Signal
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QComboBox,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    context = mp.get_context("spawn")
    shutdown_requested = threading.Event()

    def request_shutdown(
        _signal_number: int,
        _frame: Any,
    ) -> None:
        # Keep signal-handler work minimal. The Qt timer performs cleanup on
        # the GUI thread, where closing widgets and timers is safe.
        shutdown_requested.set()

    previous_sigint = signal.signal(signal.SIGINT, request_shutdown)
    previous_sigterm = signal.signal(signal.SIGTERM, request_shutdown)
    previous_sigbreak = None
    if hasattr(signal, "SIGBREAK"):
        previous_sigbreak = signal.signal(signal.SIGBREAK, request_shutdown)

    class TrainingSignals(QObject):
        progress = Signal(int)
        finished = Signal(bool, str)

    class TransactionSignals(QObject):
        loaded = Signal(object)
        failed = Signal(str)

    class TransactionDialog(QDialog):
        def __init__(self, server_url: str, parent: QWidget) -> None:
            super().__init__(parent)
            self._server_url = server_url.rstrip("/")
            self._loading = False
            self._load_thread: threading.Thread | None = None
            self._signals = TransactionSignals()
            self._signals.loaded.connect(self._show_transactions)
            self._signals.failed.connect(self._show_error)
            self.setWindowTitle("Database Transactions")
            self.resize(1200, 650)

            layout = QVBoxLayout(self)
            heading_row = QHBoxLayout()
            heading = QLabel("Inspection Transactions (newest first)")
            heading.setStyleSheet("font-size: 17px; font-weight: 700;")
            self._status = QLabel("Loading...")
            self._status.setStyleSheet("color: #8d98a8;")
            self._refresh_button = QPushButton("Refresh")
            self._refresh_button.clicked.connect(self.load_transactions)
            heading_row.addWidget(heading)
            heading_row.addWidget(self._status, 1)
            heading_row.addWidget(self._refresh_button)
            layout.addLayout(heading_row)

            headers = (
                "Timestamp",
                "Camera",
                "Event Type",
                "OCR / Captured Data",
                "Status",
                "Confidence",
                "Event ID",
            )
            self._table = QTableWidget(0, len(headers))
            self._table.setHorizontalHeaderLabels(headers)
            self._table.setEditTriggers(
                QTableWidget.EditTrigger.NoEditTriggers
            )
            self._table.setSelectionBehavior(
                QTableWidget.SelectionBehavior.SelectRows
            )
            self._table.setAlternatingRowColors(True)
            self._table.setSortingEnabled(True)
            table_header = self._table.horizontalHeader()
            table_header.setSectionResizeMode(
                QHeaderView.ResizeMode.ResizeToContents
            )
            table_header.setSectionResizeMode(
                3,
                QHeaderView.ResizeMode.Stretch,
            )
            layout.addWidget(self._table, 1)

            close_buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Close
            )
            close_buttons.rejected.connect(self.reject)
            layout.addWidget(close_buttons)
            self.load_transactions()

        def load_transactions(self) -> None:
            if self._loading:
                return
            self._loading = True
            self._refresh_button.setEnabled(False)
            self._status.setText("Loading...")
            self._load_thread = threading.Thread(
                target=self._fetch_transactions,
                name="GibraltarTransactionLoader",
                daemon=True,
            )
            self._load_thread.start()

        def _fetch_transactions(self) -> None:
            try:
                import requests

                response = requests.get(
                    f"{self._server_url}/api/inspections",
                    params={"limit": 500},
                    timeout=10,
                )
                response.raise_for_status()
                payload = response.json()
                records = payload.get("data", [])
                if not isinstance(records, list):
                    raise ValueError("Invalid transaction response from backend")
                records.sort(
                    key=lambda record: str(record.get("timestamp", "")),
                    reverse=True,
                )
                self._signals.loaded.emit(records)
            except requests.ConnectionError:
                self._signals.failed.emit(
                    "Cannot connect to the Gibraltar backend at "
                    f"{self._server_url}. Start it with: node server.js"
                )
            except requests.Timeout:
                self._signals.failed.emit(
                    "The backend did not respond in time. Please try again."
                )
            except requests.HTTPError as error:
                self._signals.failed.emit(
                    f"The backend returned HTTP {error.response.status_code}."
                )
            except (requests.RequestException, ValueError) as error:
                self._signals.failed.emit(str(error))

        def _show_transactions(self, records: object) -> None:
            self._loading = False
            self._refresh_button.setEnabled(True)
            if not isinstance(records, list):
                self._show_error("Invalid transaction data")
                return
            self._table.setSortingEnabled(False)
            self._table.setRowCount(len(records))
            for row, record in enumerate(records):
                captured_data = record.get("captured_data")
                if isinstance(captured_data, dict):
                    captured_text = str(captured_data.get("ocr_text", ""))
                elif captured_data is None:
                    captured_text = ""
                else:
                    captured_text = str(captured_data)
                values = (
                    record.get("timestamp", ""),
                    record.get("camera_id", ""),
                    record.get("event_type", ""),
                    captured_text,
                    record.get("status", ""),
                    record.get("confidence", ""),
                    record.get("event_id", ""),
                )
                for column, value in enumerate(values):
                    self._table.setItem(
                        row,
                        column,
                        QTableWidgetItem(str(value)),
                    )
            self._table.setSortingEnabled(True)
            self._table.sortItems(0, Qt.SortOrder.DescendingOrder)
            self._status.setText(f"{len(records)} transaction(s)")

        def _show_error(self, message: str) -> None:
            self._loading = False
            self._refresh_button.setEnabled(True)
            self._status.setText("Could not load transactions")
            QMessageBox.warning(self, "Database Transactions", message)

    class CameraSettingsDialog(QDialog):
        def __init__(
            self,
            camera_sources: list[dict[str, str]],
            parent: QWidget,
        ) -> None:
            super().__init__(parent)
            self.setWindowTitle("Camera Source Settings")
            self.setMinimumWidth(680)
            layout = QVBoxLayout(self)
            instruction = QLabel(
                "Choose RTSP or USB independently for each camera window, "
                "then enter its URL or USB device index."
            )
            instruction.setWordWrap(True)
            layout.addWidget(instruction)
            form = QFormLayout()
            self._source_types: list[QComboBox] = []
            self._source_fields: list[QLineEdit] = []
            for index, camera_source in enumerate(camera_sources, start=1):
                source_type = QComboBox()
                source_type.addItem("RTSP Camera", "rtsp")
                source_type.addItem("USB Camera", "usb")
                selected_type = camera_source.get("type", "rtsp")
                source_type.setCurrentIndex(1 if selected_type == "usb" else 0)
                field = QLineEdit(camera_source.get("value", ""))
                field.setClearButtonEnabled(True)
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.addWidget(source_type)
                row_layout.addWidget(field, 1)
                form.addRow(f"Camera {index} source:", row)
                self._source_types.append(source_type)
                self._source_fields.append(field)
                source_type.currentIndexChanged.connect(
                    lambda _index, combo=source_type, editor=field: (
                        self._update_source_field(combo, editor)
                    )
                )
                self._update_source_field(source_type, field)
            layout.addLayout(form)
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Save
                | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(self._validate_and_accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)

        @property
        def camera_sources(self) -> list[dict[str, str]]:
            return [
                {
                    "type": str(source_type.currentData()),
                    "value": field.text().strip(),
                }
                for source_type, field in zip(
                    self._source_types,
                    self._source_fields,
                )
            ]

        @staticmethod
        def _update_source_field(
            source_type: QComboBox,
            field: QLineEdit,
        ) -> None:
            if source_type.currentData() == "usb":
                field.setPlaceholderText("USB device index, for example 0")
            else:
                field.setPlaceholderText("rtsp://username:password@camera/stream")

        def _validate_and_accept(self) -> None:
            camera_sources = self.camera_sources
            invalid_rtsp = [
                str(index)
                for index, source in enumerate(camera_sources, start=1)
                if source["type"] == "rtsp"
                and source["value"]
                and not source["value"].lower().startswith(("rtsp://", "rtsps://"))
            ]
            if invalid_rtsp:
                QMessageBox.warning(
                    self,
                    "Invalid Camera URL",
                    "Enter an rtsp:// or rtsps:// URL for camera(s): "
                    + ", ".join(invalid_rtsp),
                )
                return
            invalid_usb = []
            for index, source in enumerate(camera_sources, start=1):
                if source["type"] != "usb" or not source["value"]:
                    continue
                try:
                    valid = int(source["value"]) >= 0
                except ValueError:
                    valid = False
                if not valid:
                    invalid_usb.append(str(index))
            if invalid_usb:
                QMessageBox.warning(
                    self,
                    "Invalid USB Camera",
                    "Enter a non-negative USB device index for camera(s): "
                    + ", ".join(invalid_usb),
                )
                return
            if not any(source["value"] for source in camera_sources):
                QMessageBox.warning(
                    self,
                    "Camera Source Required",
                    "Configure at least one RTSP or USB camera.",
                )
                return
            self.accept()

    class CameraPanel(QFrame):
        match_detected = Signal(str, float, str, object)

        def __init__(
            self,
            title: str,
            camera_name: str,
            repository_root: Path,
            match_event_cooldown: float,
        ) -> None:
            super().__init__()
            self._camera_name = camera_name
            self._snapshot_dir = repository_root / camera_name
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)
            self._latest_jpeg: bytes | None = None
            self._latest_metadata: dict[str, Any] = {}
            self._model_feature: np.ndarray | None = None
            self._match_threshold = 0.75
            self._match_frame_counter = 0
            self._match_event_cooldown = match_event_cooldown
            self._last_match_event_time = 0.0
            self._previously_matched = False
            self._ocr_collection_started: float | None = None
            self._best_ocr_metadata: dict[str, Any] = {}
            self._best_ocr_text = ""
            self._training_thread: threading.Thread | None = None
            self._training_signals = TrainingSignals()
            self._training_signals.progress.connect(
                self._set_training_progress
            )
            self._training_signals.finished.connect(
                self._training_finished
            )
            self.setFrameShape(QFrame.Shape.StyledPanel)
            self.setStyleSheet(
                "QFrame { background: #15191f; border: 1px solid #343b46; }"
                "QLabel { color: #e7edf5; border: none; }"
            )
            layout = QVBoxLayout(self)
            heading = QLabel(title)
            heading.setStyleSheet("font-size: 16px; font-weight: 600;")
            self.video = QLabel("Starting...")
            self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.video.setMinimumSize(420, 260)
            self.video.setStyleSheet("background: #090b0e; color: #8d98a8;")
            self.status = QLabel("Loading pipeline...")
            self.status.setStyleSheet("color: #8d98a8;")
            self.save_button = QPushButton("Save")
            self.save_button.setEnabled(False)
            self.save_button.setCursor(
                Qt.CursorShape.PointingHandCursor,
            )
            self.save_button.setStyleSheet(
                "QPushButton {"
                " background: #2478d4; color: white; border: none;"
                " border-radius: 5px; padding: 9px; font-weight: 600;"
                "}"
                "QPushButton:hover { background: #3389e6; }"
                "QPushButton:pressed { background: #1b62b0; }"
                "QPushButton:disabled {"
                " background: #343b46; color: #7b8491;"
                "}"
            )
            self.save_button.clicked.connect(self.save_snapshot)
            self.train_button = QPushButton("Train Model")
            self.train_button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.train_button.setStyleSheet(self.save_button.styleSheet())
            self.train_button.clicked.connect(self.train_model)
            self.training_progress = QProgressBar()
            self.training_progress.setRange(0, 100)
            self.training_progress.setValue(0)
            self.training_progress.setTextVisible(True)
            self.training_progress.setVisible(False)
            self.match_badge = QLabel("MODEL NOT TRAINED")
            self.match_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.match_badge.setStyleSheet(
                "color: #f4b942; font-size: 15px; font-weight: 700;"
            )
            layout.addWidget(heading)
            layout.addWidget(self.video, 1)
            layout.addWidget(self.match_badge)
            layout.addWidget(self.status)
            layout.addWidget(self.save_button)
            layout.addWidget(self.train_button)
            layout.addWidget(self.training_progress)
            self._load_model()

        def set_repository_root(self, repository_root: Path) -> None:
            self._snapshot_dir = repository_root / self._camera_name
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)
            self._model_feature = None
            self._load_model()

        @property
        def _model_path(self) -> Path:
            return self._snapshot_dir / "visual_match_model.npz"

        def _load_model(self) -> None:
            try:
                with np.load(self._model_path) as model:
                    self._model_feature = model["feature"].astype(np.float32)
                    self._match_threshold = float(model["threshold"])
                self.match_badge.setText("MODEL READY")
                self.match_badge.setStyleSheet(
                    "color: #59c3ff; font-size: 15px; font-weight: 700;"
                )
            except (OSError, KeyError, ValueError):
                self._model_feature = None
                self.match_badge.setText("MODEL NOT TRAINED")
                self.match_badge.setStyleSheet(
                    "color: #f4b942; font-size: 15px; font-weight: 700;"
                )

        def show_message(self, text: str, *, error: bool = False) -> None:
            self.status.setText(text)
            self.status.setStyleSheet(
                "color: #ff6b6b;" if error else "color: #8d98a8;"
            )

        def show_jpeg(
            self,
            data: bytes,
            metadata: dict[str, Any] | None = None,
        ) -> None:
            image = QImage.fromData(data, "JPG")
            if image.isNull():
                self.show_message("Invalid frame received", error=True)
                return
            self._latest_jpeg = data
            self._latest_metadata = metadata or {}
            self.save_button.setEnabled(True)
            self.video.setPixmap(
                QPixmap.fromImage(image).scaled(
                    self.video.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self.status.setText(f"Live | {image.width()}x{image.height()}")
            self._match_frame_counter += 1
            if (
                self._model_feature is not None
                and self._match_frame_counter % 5 == 0
            ):
                frame = decode_frame(data)
                feature = extract_visual_feature(frame)
                similarity = visual_similarity(
                    feature,
                    self._model_feature,
                )
                if similarity >= self._match_threshold:
                    self.match_badge.setText(
                        f"\u2714 MATCHED ({similarity:.0%})"
                    )
                    self.match_badge.setStyleSheet(
                        "color: #35d07f; font-size: 17px; font-weight: 800;"
                    )
                    now = time.monotonic()
                    if (
                        not self._previously_matched
                        or now - self._last_match_event_time
                        >= self._match_event_cooldown
                    ):
                        self._last_match_event_time = now
                        self.match_detected.emit(
                            self._camera_name,
                            similarity,
                            str(self._model_path),
                            self._latest_metadata,
                        )
                    self._previously_matched = True
                else:
                    self.match_badge.setText(
                        f"\u2716 NOT MATCHED ({similarity:.0%})"
                    )
                    self.match_badge.setStyleSheet(
                        "color: #ff5f65; font-size: 17px; font-weight: 800;"
                    )
                    self._previously_matched = False
                    self._ocr_collection_started = None
                    self._best_ocr_metadata = {}
                    self._best_ocr_text = ""

        def save_snapshot(self) -> None:
            if self._latest_jpeg is None:
                self.show_message("No frame available to save", error=True)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            snapshot_path = (
                self._snapshot_dir
                / f"{self._camera_name}_{timestamp}.jpg"
            )
            try:
                snapshot_path.write_bytes(self._latest_jpeg)
            except OSError as error:
                self.show_message(
                    f"Save failed: {error}",
                    error=True,
                )
                return
            self.show_message(f"Saved: {snapshot_path.name}")

        def train_model(self) -> None:
            if (
                self._training_thread is not None
                and self._training_thread.is_alive()
            ):
                return
            image_paths = sorted(
                path
                for path in self._snapshot_dir.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            )
            if not image_paths:
                self.show_message(
                    "Save at least one image before training",
                    error=True,
                )
                return
            self.train_button.setEnabled(False)
            self.training_progress.setValue(0)
            self.training_progress.setVisible(True)
            self.match_badge.setText("TRAINING...")
            self.match_badge.setStyleSheet(
                "color: #f4b942; font-size: 15px; font-weight: 700;"
            )
            self._training_thread = threading.Thread(
                target=self._train_model_worker,
                args=(image_paths,),
                name=f"{self._camera_name}-visual-training",
                daemon=True,
            )
            self._training_thread.start()

        def _train_model_worker(self, image_paths: list[Path]) -> None:
            try:
                features: list[np.ndarray] = []
                total = len(image_paths)
                for index, image_path in enumerate(image_paths, start=1):
                    frame = cv2.imread(str(image_path))
                    if frame is not None:
                        features.append(extract_visual_feature(frame))
                    self._training_signals.progress.emit(
                        round(index * 90 / total)
                    )
                if not features:
                    raise ValueError("No readable training images found")
                model_feature = np.mean(features, axis=0).astype(np.float32)
                cv2.normalize(model_feature, model_feature)
                if len(features) > 1:
                    training_scores = [
                        visual_similarity(feature, model_feature)
                        for feature in features
                    ]
                    threshold = max(0.55, min(training_scores) - 0.10)
                else:
                    threshold = 0.75
                np.savez_compressed(
                    self._model_path,
                    feature=model_feature,
                    threshold=np.float32(threshold),
                    image_count=np.int32(len(features)),
                )
                self._training_signals.progress.emit(100)
                self._training_signals.finished.emit(
                    True,
                    f"Trained with {len(features)} image(s)",
                )
            except (OSError, ValueError, cv2.error) as error:
                self._training_signals.finished.emit(False, str(error))

        def _set_training_progress(self, progress: int) -> None:
            self.training_progress.setValue(progress)

        def _training_finished(self, success: bool, message: str) -> None:
            self.train_button.setEnabled(True)
            if success:
                self._load_model()
                self.show_message(message)
            else:
                self.match_badge.setText("TRAINING FAILED")
                self.match_badge.setStyleSheet(
                    "color: #ff5f65; font-size: 15px; font-weight: 700;"
                )
                self.show_message(f"Training failed: {message}", error=True)

    class GibraltarWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Gibraltar Visual AI")
            self.resize(1450, 620)
            self._stopping = False
            self._stop_event = context.Event()
            project_config = load_project_config()
            database_config = project_config.get("database", {})
            self._database_name = str(
                database_config.get("name", "gibraltar")
            )
            self._database_table = str(
                database_config.get("table", "inspection_records")
            )
            self._match_event_cooldown = max(
                1.0,
                float(database_config.get("match_event_cooldown_seconds", 30)),
            )
            server_url = str(
                database_config.get(
                    "server_url",
                    os.getenv("SOCKET_SERVER_URL", "http://127.0.0.1:3000"),
                )
            )
            self._server_url = server_url
            self._socket_client = SocketClient(server_url)
            self._inspection_sender = InspectionSender(
                self._socket_client,
                site_id=str(database_config.get("site_id", "GIBRALTAR")),
                default_section_id=str(
                    database_config.get("section_id", "VISUAL-AI")
                ),
            )
            self._database_connect_thread = threading.Thread(
                target=self._socket_client.connect,
                name="GibraltarDatabaseConnection",
                daemon=True,
            )
            self._database_connect_thread.start()
            self._send_startup_test_transactions()
            self._camera_sources = load_camera_sources()
            self._cameras: list[RTSPCamera | LocalCamera | None] = [None, None, None]

            root = QWidget()
            root.setStyleSheet("background: #0f1217;")
            layout = QGridLayout(root)
            self.setCentralWidget(root)
            self._repository_root = load_image_repository_path()
            self._repository_root.mkdir(parents=True, exist_ok=True)

            repository_bar = QWidget()
            repository_layout = QHBoxLayout(repository_bar)
            repository_layout.setContentsMargins(0, 0, 0, 4)
            repository_title = QLabel("Snapshot folder:")
            repository_title.setStyleSheet("color: #e7edf5; font-weight: 600;")
            self._repository_path_label = QLabel(str(self._repository_root))
            self._repository_path_label.setStyleSheet("color: #9ca8b8;")
            select_repository_button = QPushButton("Select Save Folder")
            select_repository_button.setStyleSheet(
                "QPushButton {"
                " background: #2478d4; color: white; border: none;"
                " border-radius: 5px; padding: 8px 14px; font-weight: 600;"
                "}"
                "QPushButton:hover { background: #3389e6; }"
            )
            select_repository_button.clicked.connect(
                self._select_repository_path
            )
            camera_settings_button = QPushButton("Select Camera Source")
            camera_settings_button.setStyleSheet(
                select_repository_button.styleSheet()
            )
            camera_settings_button.clicked.connect(
                self._open_camera_settings
            )
            transactions_button = QPushButton("View Transactions")
            transactions_button.setStyleSheet(
                select_repository_button.styleSheet()
            )
            transactions_button.clicked.connect(
                self._open_transactions
            )
            repository_layout.addWidget(repository_title)
            repository_layout.addWidget(self._repository_path_label, 1)
            repository_layout.addWidget(select_repository_button)
            repository_layout.addWidget(camera_settings_button)
            repository_layout.addWidget(transactions_button)
            layout.addWidget(repository_bar, 0, 0, 1, 3)

            specifications = (
                ("OCR Camera", "ocr", args.ocr_device),
                ("Object Detection Camera 1", "object", args.object_device),
                ("Object Detection Camera 2", "object", args.object_device),
            )
            self._panels: list[CameraPanel] = []
            self._input_queues: list[Any] = []
            self._output_queues: list[Any] = []
            self._workers: list[Any] = []

            for column, (title, kind, device) in enumerate(specifications):
                camera_name = f"camera-{column + 1}"
                panel = CameraPanel(
                    title,
                    camera_name,
                    self._repository_root,
                    self._match_event_cooldown,
                )
                panel.match_detected.connect(self._record_camera_match)
                layout.addWidget(panel, 1, column)
                self._panels.append(panel)
                input_queue = context.Queue(maxsize=1)
                output_queue = context.Queue(maxsize=2)
                worker = context.Process(
                    target=processing_worker,
                    args=(
                        kind,
                        input_queue,
                        output_queue,
                        self._stop_event,
                        device,
                    ),
                    name=f"Gibraltar-{kind}-{column + 1}",
                    daemon=True,
                )
                worker.start()
                self._input_queues.append(input_queue)
                self._output_queues.append(output_queue)
                self._workers.append(worker)

            self._connect_cameras()

            self._timer = QTimer(self)
            self._timer.timeout.connect(self._update)
            self._timer.start(max(1, round(1000 / args.fps)))

        def _send_startup_test_transactions(self) -> None:
            """Send ten pipeline-test records once when this Edge process starts."""
            for index in range(1, 11):
                self._inspection_sender.send(
                    camera_id="EDGE-STARTUP-TEST",
                    section_id="PIPELINE-TEST",
                    event_type="edge_startup_pipeline_test",
                    status="PASS",
                    captured_data=f"DUMMY-TRANSACTION-{index:02d}",
                    confidence=1.0,
                    comments="Dummy Edge-to-database pipeline test",
                    remarks=f"Startup test record {index} of 10",
                )

        def _record_camera_match(
            self,
            camera_name: str,
            similarity: float,
            model_path: str,
            metadata: object,
        ) -> None:
            """Save a positive trained-model match from any camera."""
            ocr_data: dict[str, Any] = {}
            if isinstance(metadata, dict):
                candidate = metadata.get("ocr", {})
                if isinstance(candidate, dict):
                    ocr_data = candidate
            captured_text = " | ".join(
                line.strip()
                for line in str(ocr_data.get("text", "")).splitlines()
                if line.strip()
            )
            captured_data: dict[str, Any] = {
                "matched": True,
                "similarity": round(similarity, 4),
                "model_path": model_path,
            }
            if captured_text:
                captured_data["ocr_text"] = captured_text
            result = self._inspection_sender.send_pass(
                camera_id=camera_name,
                section_id=camera_name.upper(),
                event_type="trained_model_match",
                captured_data=captured_data,
                confidence=round(similarity, 4),
                evidence_link=model_path,
                comments=f"{camera_name} frame matched the trained model",
                remarks="Automatically generated by Gibraltar Visual AI",
            )
            panel_index = int(camera_name.rsplit("-", 1)[-1]) - 1
            if 0 <= panel_index < len(self._panels):
                if result.get("success"):
                    self._panels[panel_index].show_message(
                        f"Match saved to database ({result['event_id']})"
                    )
                else:
                    self._panels[panel_index].show_message(
                        "Match queued; waiting for database connection",
                        error=True,
                    )

        def _select_repository_path(self) -> None:
            selected = QFileDialog.getExistingDirectory(
                self,
                "Select snapshot repository",
                str(self._repository_root),
            )
            if not selected:
                return
            repository_root = Path(selected).resolve()
            try:
                repository_root.mkdir(parents=True, exist_ok=True)
                save_image_repository_path(repository_root)
                for panel in self._panels:
                    panel.set_repository_root(repository_root)
            except OSError as error:
                for panel in self._panels:
                    panel.show_message(
                        f"Could not use save folder: {error}",
                        error=True,
                    )
                return
            self._repository_root = repository_root
            self._repository_path_label.setText(str(repository_root))

        def _open_camera_settings(self) -> None:
            dialog = CameraSettingsDialog(self._camera_sources, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            camera_sources = dialog.camera_sources
            try:
                save_camera_sources(camera_sources)
            except OSError as error:
                QMessageBox.critical(
                    self,
                    "Camera Settings",
                    f"Could not save camera sources: {error}",
                )
                return
            self._camera_sources = camera_sources
            self._connect_cameras()

        def _open_transactions(self) -> None:
            dialog = TransactionDialog(self._server_url, self)
            dialog.exec()

        def _connect_cameras(self) -> None:
            for camera in self._cameras:
                if camera is not None:
                    camera.release()
            self._cameras = [None, None, None]

            for index, (camera_source, panel) in enumerate(
                zip(self._camera_sources, self._panels)
            ):
                panel.video.clear()
                panel.save_button.setEnabled(False)
                source_value = camera_source["value"]
                if not source_value:
                    panel.video.setText(f"Camera {index + 1} not configured")
                    panel.show_message(
                        "Use Select Camera Source to configure this camera",
                        error=True,
                    )
                    continue
                panel.video.setText(f"Connecting Camera {index + 1}...")
                if camera_source["type"] == "usb":
                    camera = LocalCamera(
                        int(source_value),
                        width=args.width,
                        height=args.height,
                        fps=args.fps,
                    )
                    source_label = f"USB {source_value}"
                else:
                    camera = RTSPCamera(source_value)
                    source_label = "RTSP"
                self._cameras[index] = camera
                try:
                    camera.open()
                except ConnectionError as error:
                    panel.video.setText(f"Camera {index + 1} unavailable")
                    panel.show_message(str(error), error=True)
                else:
                    panel.show_message(
                        f"Camera {index + 1} ({source_label}) connected; waiting for frame"
                    )

        def _update(self) -> None:
            if shutdown_requested.is_set():
                self.close()
                return

            for camera, panel, input_queue in zip(
                self._cameras,
                self._panels,
                self._input_queues,
            ):
                if camera is None:
                    continue
                ok, frame = camera.read_latest()
                if ok:
                    encoded = encode_frame(frame)
                    latest_put(input_queue, encoded)

            for panel, output_queue, worker in zip(
                self._panels,
                self._output_queues,
                self._workers,
            ):
                latest = None
                while True:
                    try:
                        latest = output_queue.get_nowait()
                    except queue.Empty:
                        break
                if latest is not None:
                    message_type, payload = latest
                    if message_type == "frame":
                        frame_data, metadata = payload
                        panel.show_jpeg(frame_data, metadata)
                    else:
                        panel.show_message(
                            payload,
                            error=message_type == "error",
                        )
                elif not worker.is_alive() and worker.exitcode is not None:
                    panel.show_message(
                        f"Worker stopped (exit code {worker.exitcode})",
                        error=True,
                    )

        def shutdown(self) -> None:
            if self._stopping:
                return
            self._stopping = True
            self._timer.stop()
            for camera in self._cameras:
                if camera is not None:
                    camera.release()
            self._stop_event.set()
            self._socket_client.disconnect()

            # Wake workers immediately instead of waiting for Queue.get()
            # timeouts. Old queued frames are discarded by latest_put.
            for input_queue in self._input_queues:
                latest_put(input_queue, None)

            for worker in self._workers:
                worker.join(timeout=8.0)
                if worker.is_alive():
                    print(
                        f"Stopping unresponsive worker: {worker.name}",
                        file=sys.stderr,
                    )
                    worker.terminate()
                    worker.join(timeout=3.0)

            for pipeline_queue in (
                *self._input_queues,
                *self._output_queues,
            ):
                # Do not wait on multiprocessing feeder threads during
                # interpreter shutdown; all consumers are already stopped.
                pipeline_queue.cancel_join_thread()
                pipeline_queue.close()

        def closeEvent(self, event: Any) -> None:
            self.shutdown()
            event.accept()

    application = QApplication(sys.argv)
    window = GibraltarWindow()
    application.aboutToQuit.connect(window.shutdown)
    window.showMaximized()
    try:
        return application.exec()
    except KeyboardInterrupt:
        shutdown_requested.set()
        window.close()
        return 0
    finally:
        window.shutdown()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if previous_sigbreak is not None:
            signal.signal(signal.SIGBREAK, previous_sigbreak)


def main() -> int:
    try:
        return run_gui(parse_args())
    except ImportError as error:
        print(f"Missing GUI dependency: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
