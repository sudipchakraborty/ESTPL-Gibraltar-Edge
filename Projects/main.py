"""Gibraltar three-panel visual AI dashboard using three camera sources.

Each configured RTSP or USB camera feeds its respective OCR/object-detection
worker and panel. Sources are managed from the GUI and persisted in config.json.
"""
####.\.venv\Scripts\python.exe Projects/main.py

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import unicodedata
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
    default_path = REPOSITORY_DIR / "runtime" / "Saved_Image_Gibraltar"
    configured = load_project_config().get("image_repository_path")
    if configured:
        expanded = os.path.expandvars(os.path.expanduser(str(configured)))
        configured_path = Path(expanded)
        if not configured_path.is_absolute():
            configured_path = REPOSITORY_DIR / configured_path
        return configured_path.resolve()
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


def load_camera_rois() -> dict[str, list[float] | None]:
    config = load_project_config()
    configured = config.get("camera_rois", {})
    rois: dict[str, list[float] | None] = {}
    for index in range(1, 4):
        value = configured.get(f"camera-{index}") if isinstance(configured, dict) else None
        if isinstance(value, list) and len(value) == 4:
            try:
                x, y, width, height = (float(item) for item in value)
            except (TypeError, ValueError):
                value = None
            else:
                if width > 0 and height > 0:
                    rois[f"camera-{index}"] = [
                        max(0.0, min(1.0, x)),
                        max(0.0, min(1.0, y)),
                        max(0.0, min(1.0 - x, width)),
                        max(0.0, min(1.0 - y, height)),
                    ]
                    continue
        rois[f"camera-{index}"] = None
    return rois


def save_camera_rois(rois: dict[str, list[float] | None]) -> None:
    config = load_project_config()
    config["camera_rois"] = rois
    temporary_path = CONFIG_PATH.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=4)
        config_file.write("\n")
    temporary_path.replace(CONFIG_PATH)


def crop_to_roi(frame: np.ndarray, roi: list[float] | None) -> np.ndarray:
    if not roi:
        return frame
    frame_height, frame_width = frame.shape[:2]
    x, y, width, height = roi
    left = max(0, min(frame_width - 1, round(x * frame_width)))
    top = max(0, min(frame_height - 1, round(y * frame_height)))
    right = max(left + 1, min(frame_width, round((x + width) * frame_width)))
    bottom = max(top + 1, min(frame_height, round((y + height) * frame_height)))
    return frame[top:bottom, left:right]


def image_sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def roi_object_difference(
    frame: np.ndarray,
    empty_frame: np.ndarray | None,
) -> tuple[bool, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if empty_frame is None or empty_frame.shape != gray.shape:
        return False, gray
    difference = cv2.absdiff(gray, empty_frame)
    changed_pixels = float(np.mean(difference > 18))
    mean_difference = float(difference.mean())
    return changed_pixels >= 0.08 or mean_difference >= 10.0, gray


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


def draw_ocr_overlay(
    frame: np.ndarray,
    metadata: dict[str, Any],
    roi: list[float] | None = None,
) -> np.ndarray:
    """Paint the latest OCR result on a fresh live frame."""
    output = frame.copy()
    ocr = metadata.get("ocr", {})
    if not isinstance(ocr, dict):
        return output
    source_width = max(1, int(ocr.get("image_width", output.shape[1])))
    source_height = max(1, int(ocr.get("image_height", output.shape[0])))
    roi_left = roi[0] * output.shape[1] if roi else 0.0
    roi_top = roi[1] * output.shape[0] if roi else 0.0
    roi_width = roi[2] * output.shape[1] if roi else output.shape[1]
    roi_height = roi[3] * output.shape[0] if roi else output.shape[0]
    scale_x = roi_width / source_width
    scale_y = roi_height / source_height
    words = ocr.get("words", [])
    if isinstance(words, list):
        for word in words:
            if not isinstance(word, dict):
                continue
            box = word.get("box", [])
            try:
                if len(box) == 4 and not isinstance(box[0], (list, tuple)):
                    x1, y1, x2, y2 = (int(value) for value in box)
                else:
                    points = [(int(point[0]), int(point[1])) for point in box]
                    if not points:
                        continue
                    xs, ys = zip(*points)
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            except (TypeError, ValueError, IndexError):
                continue
            x1, x2 = int(roi_left + x1 * scale_x), int(roi_left + x2 * scale_x)
            y1, y2 = int(roi_top + y1 * scale_y), int(roi_top + y2 * scale_y)
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = str(word.get("text", ""))
            confidence = float(word.get("confidence", 0.0))
            cv2.putText(
                output,
                f"{text} ({confidence:.2f})",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
    inference_ms = float(ocr.get("inference_time_ms", 0.0))
    cv2.putText(
        output,
        f"OCR {inference_ms:.0f} ms (async)",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def create_ocr_processor(
    device: str,
) -> Callable[[np.ndarray], dict[str, Any]]:
    from OCR.PaddleOCR import OCR, OCRConfig

    reader = OCR(
        OCRConfig(
            device=device,
            detection_limit_side_len=640,
            recognition_batch_size=8,
        )
    )

    def process(frame: np.ndarray) -> dict[str, Any]:
        # Local contrast enhancement helps expose shallow embossed strokes.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        result = reader.read(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))
        return {"ocr": result.as_dict()}

    return process


def create_detection_processor(
    device: str,
) -> Callable[[np.ndarray], dict[str, Any]]:
    from models.detector import ObjectDetector

    detector = ObjectDetector(
        str(REPOSITORY_DIR / "models" / "weights" / "yolov8n.pt"),
        device=device,
    )

    def process(frame: np.ndarray) -> dict[str, Any]:
        started = time.perf_counter()
        detections, _results = detector.detect(frame)
        height, width = frame.shape[:2]
        valid = []
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            valid.append({
                **detection,
                "bbox": [x1, y1, x2, y2],
                "feature": extract_visual_feature(frame[y1:y2, x1:x2]).tolist(),
            })
        return {"objects": valid, "image_width": width, "image_height": height,
                "inference_time_ms": (time.perf_counter() - started) * 1000.0}

    return process


def draw_object_overlay(frame: np.ndarray, metadata: dict[str, Any], roi: list[float] | None) -> np.ndarray:
    output = frame.copy()
    height, width = frame.shape[:2]
    left, top, roi_width, roi_height = roi or [0.0, 0.0, 1.0, 1.0]
    sx = width * roi_width / max(1, int(metadata.get("image_width", width)))
    sy = height * roi_height / max(1, int(metadata.get("image_height", height)))
    for obj in metadata.get("objects", []):
        x1, y1, x2, y2 = obj["bbox"]
        start = (int(width * left + x1 * sx), int(height * top + y1 * sy))
        end = (int(width * left + x2 * sx), int(height * top + y2 * sy))
        cv2.rectangle(output, start, end, (0, 220, 255), 2)
        cv2.putText(output, f"{obj['class_name']} {obj['confidence']:.0%}",
                    (start[0], max(18, start[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 220, 255), 2, cv2.LINE_AA)
    return output


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
            latest_put(output_queue, ("analysis", processed))
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
    from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
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
    from audio_alerts import AudioAlertConfigStore
    from audio_alerts.qt import AudioAlertConfigDialog, AudioAlertManager

    class RoiVideoLabel(QLabel):
        roi_changed = Signal(object)

        def __init__(self, roi: list[float] | None = None) -> None:
            super().__init__()
            self._roi = roi
            self._drawing = False
            self._start = None
            self._current = None
            self.setMouseTracking(True)

        @property
        def roi(self) -> list[float] | None:
            return list(self._roi) if self._roi else None

        def set_roi(self, roi: list[float] | None) -> None:
            self._roi = list(roi) if roi else None
            self.update()

        def set_drawing(self, enabled: bool) -> None:
            self._drawing = enabled
            self._start = None
            self._current = None
            self.setCursor(
                Qt.CursorShape.CrossCursor
                if enabled
                else Qt.CursorShape.ArrowCursor
            )
            self.update()

        def _image_rect(self):
            pixmap = self.pixmap()
            if pixmap is None or pixmap.isNull():
                return None
            width = pixmap.width()
            height = pixmap.height()
            return (
                (self.width() - width) / 2,
                (self.height() - height) / 2,
                width,
                height,
            )

        def _normalized_point(self, point):
            image_rect = self._image_rect()
            if image_rect is None:
                return None
            left, top, width, height = image_rect
            x = max(0.0, min(1.0, (point.x() - left) / width))
            y = max(0.0, min(1.0, (point.y() - top) / height))
            return x, y

        def mousePressEvent(self, event) -> None:
            if self._drawing and event.button() == Qt.MouseButton.LeftButton:
                self._start = self._normalized_point(event.position().toPoint())
                self._current = self._start
                self.update()
                return
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event) -> None:
            if self._drawing and self._start is not None:
                self._current = self._normalized_point(event.position().toPoint())
                self.update()
                return
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event) -> None:
            if (
                self._drawing
                and self._start is not None
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self._current = self._normalized_point(event.position().toPoint())
                if self._current is not None:
                    start_x, start_y = self._start
                    end_x, end_y = self._current
                    left, right = sorted((start_x, end_x))
                    top, bottom = sorted((start_y, end_y))
                    if right - left >= 0.01 and bottom - top >= 0.01:
                        self._roi = [left, top, right - left, bottom - top]
                        self.roi_changed.emit(self.roi)
                self._start = None
                self._current = None
                self.update()
                return
            super().mouseReleaseEvent(event)

        def paintEvent(self, event) -> None:
            super().paintEvent(event)
            roi = self._roi
            if self._drawing and self._start and self._current:
                start_x, start_y = self._start
                end_x, end_y = self._current
                roi = [
                    min(start_x, end_x),
                    min(start_y, end_y),
                    abs(end_x - start_x),
                    abs(end_y - start_y),
                ]
            image_rect = self._image_rect()
            if not roi or image_rect is None:
                return
            left, top, width, height = image_rect
            painter = QPainter(self)
            painter.setPen(QPen(QColor("#ffd166"), 3))
            painter.setBrush(QColor(255, 209, 102, 45))
            painter.drawRect(
                round(left + roi[0] * width),
                round(top + roi[1] * height),
                round(roi[2] * width),
                round(roi[3] * height),
            )
            painter.end()

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

    class CameraDiscoverySignals(QObject):
        completed = Signal(object)

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
            self._rtsp_fields: list[QLineEdit] = []
            self._usb_fields: list[QComboBox] = []
            self._usb_containers: list[QWidget] = []
            self._discovery_running = False
            self._discovery_signals = CameraDiscoverySignals()
            self._discovery_signals.completed.connect(
                self._populate_usb_cameras
            )
            for index, camera_source in enumerate(camera_sources, start=1):
                source_type = QComboBox()
                source_type.addItem("RTSP Camera", "rtsp")
                source_type.addItem("USB Camera", "usb")
                selected_type = camera_source.get("type", "rtsp")
                source_type.setCurrentIndex(1 if selected_type == "usb" else 0)
                rtsp_field = QLineEdit(
                    camera_source.get("value", "")
                    if selected_type == "rtsp"
                    else ""
                )
                rtsp_field.setClearButtonEnabled(True)
                rtsp_field.setPlaceholderText(
                    "rtsp://username:password@camera/stream"
                )
                usb_field = QComboBox()
                usb_field.setMinimumWidth(220)
                usb_value = (
                    camera_source.get("value", "")
                    if selected_type == "usb"
                    else ""
                )
                if usb_value:
                    usb_field.addItem(f"USB Camera {usb_value}", usb_value)
                else:
                    usb_field.addItem("Select a USB camera", "")
                refresh_button = QPushButton("Refresh")
                refresh_button.clicked.connect(self._refresh_usb_cameras)
                usb_container = QWidget()
                usb_layout = QHBoxLayout(usb_container)
                usb_layout.setContentsMargins(0, 0, 0, 0)
                usb_layout.addWidget(usb_field, 1)
                usb_layout.addWidget(refresh_button)
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.addWidget(source_type)
                row_layout.addWidget(rtsp_field, 1)
                row_layout.addWidget(usb_container, 1)
                form.addRow(f"Camera {index} source:", row)
                self._source_types.append(source_type)
                self._rtsp_fields.append(rtsp_field)
                self._usb_fields.append(usb_field)
                self._usb_containers.append(usb_container)
                source_type.currentIndexChanged.connect(
                    lambda _index, combo=source_type,
                    rtsp=rtsp_field, usb=usb_container: (
                        self._source_type_changed(combo, rtsp, usb)
                    )
                )
                self._update_source_widgets(
                    source_type,
                    rtsp_field,
                    usb_container,
                )
            layout.addLayout(form)
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Save
                | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(self._validate_and_accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)
            if any(
                source_type.currentData() == "usb"
                for source_type in self._source_types
            ):
                self._refresh_usb_cameras()

        @property
        def camera_sources(self) -> list[dict[str, str]]:
            sources: list[dict[str, str]] = []
            for source_type, rtsp_field, usb_field in zip(
                self._source_types,
                self._rtsp_fields,
                self._usb_fields,
            ):
                kind = str(source_type.currentData())
                value = (
                    str(usb_field.currentData() or "")
                    if kind == "usb"
                    else rtsp_field.text().strip()
                )
                sources.append({"type": kind, "value": value})
            return sources

        def _source_type_changed(
            self,
            source_type: QComboBox,
            rtsp_field: QLineEdit,
            usb_container: QWidget,
        ) -> None:
            self._update_source_widgets(
                source_type,
                rtsp_field,
                usb_container,
            )
            if source_type.currentData() == "usb":
                self._refresh_usb_cameras()

        @staticmethod
        def _update_source_widgets(
            source_type: QComboBox,
            rtsp_field: QLineEdit,
            usb_container: QWidget,
        ) -> None:
            usb_selected = source_type.currentData() == "usb"
            rtsp_field.setVisible(not usb_selected)
            usb_container.setVisible(usb_selected)

        def _refresh_usb_cameras(self) -> None:
            if self._discovery_running:
                return
            self._discovery_running = True
            for usb_field in self._usb_fields:
                previous_value = str(usb_field.currentData() or "")
                usb_field.clear()
                usb_field.addItem("Scanning USB cameras...", previous_value)
                usb_field.setEnabled(False)
            threading.Thread(
                target=self._discover_usb_cameras,
                name="GibraltarUsbCameraDiscovery",
                daemon=True,
            ).start()

        def _discover_usb_cameras(self) -> None:
            from camera.usb_discovery import discover_usb_cameras

            devices = discover_usb_cameras()
            self._discovery_signals.completed.emit(devices)

        def _populate_usb_cameras(self, devices: object) -> None:
            self._discovery_running = False
            discovered = devices if isinstance(devices, list) else []
            for usb_field in self._usb_fields:
                previous_value = str(usb_field.currentData() or "")
                usb_field.clear()
                if not discovered:
                    usb_field.addItem("No USB cameras found", "")
                else:
                    for device in discovered:
                        if not isinstance(device, dict):
                            continue
                        usb_field.addItem(
                            str(device.get("label", "USB Camera")),
                            str(device.get("index", "")),
                        )
                    selected_index = usb_field.findData(previous_value)
                    if selected_index >= 0:
                        usb_field.setCurrentIndex(selected_index)
                usb_field.setEnabled(True)

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
        inspection_completed = Signal(str, bool, float, str, object)
        classification_changed = Signal(str, bool)
        inspection_reset = Signal(str)

        def __init__(
            self,
            title: str,
            camera_name: str,
            repository_root: Path,
            match_event_cooldown: float,
            roi: list[float] | None,
        ) -> None:
            super().__init__()
            self._camera_name = camera_name
            self._is_ocr = camera_name == "camera-1"
            # Camera 1 keeps the automatic OCR/reference workflow. The two
            # object cameras use the operator-driven snapshot/training flow.
            self._automatic_detection = self._is_ocr
            self._object_references: list[dict[str, Any]] = []
            self._text_references: list[str] = []
            self._pending_reference_frame = None
            self._active_reference_capture = False
            self._ocr_frame = None
            self._candidate_reference = ""
            self._last_result: bool | None = None
            self._confident_match_readings = 0
            self._required_match_readings = 3
            self._removal_readings = 0
            self._removal_started: float | None = None
            self._required_removal_readings = 5
            self._removal_hold_seconds = 1.5
            inspection_config = load_project_config().get(f"{camera_name.replace('-', '_')}_inspection", {})
            if not isinstance(inspection_config, dict):
                inspection_config = {}
            def number_setting(name: str, default: float, low: float, high: float) -> float:
                try:
                    value = float(inspection_config.get(name, default))
                    return value if low <= value <= high else default
                except (TypeError, ValueError):
                    return default
            self._placement_wait_seconds = number_setting("placement_wait_seconds", 2.0, 0.0, 60.0)
            self._automatic_match_threshold = number_setting("match_score_threshold", 0.85, 0.0, 1.0)
            self._ocr_confidence_threshold = number_setting(
                "ocr_confidence_threshold" if self._is_ocr else "object_confidence_threshold",
                0.85 if self._is_ocr else 0.60, 0.0, 1.0)
            self._placement_confirm_readings = int(number_setting("placement_confirm_readings", 3, 2, 30))
            self._placement_confirm_seconds = number_setting("placement_confirm_seconds", 0.6, 0.1, 10.0)
            self._placement_min_text_length = int(number_setting("placement_min_text_length", 2, 1, 100))
            self._presence_text = ""
            self._presence_readings = 0
            self._presence_started = None
            self._placement_missing_readings = 0
            self._placement_started: float | None = None
            self._window_best_score = 0.0
            self._window_best_metadata: dict[str, Any] = {}
            self._window_has_confident_text = False
            self._snapshot_dir = repository_root / camera_name
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)
            self._latest_jpeg: bytes | None = None
            self._latest_inspection_frame: np.ndarray | None = None
            self._latest_metadata: dict[str, Any] = {}
            self._model_feature: np.ndarray | None = None
            self._match_threshold = 0.75
            self._match_frame_counter = 0
            self._match_event_cooldown = match_event_cooldown
            self._last_match_event_time = 0.0
            self._previously_matched = False
            self._last_audio_classification: bool | None = None
            self._display_frame_count = 0
            self._display_fps_started = time.monotonic()
            self._display_fps = 0.0
            self._ocr_collection_started: float | None = None
            self._best_ocr_metadata: dict[str, Any] = {}
            self._best_ocr_text = ""
            self._inspection_locked = False
            self._awaiting_ocr = False
            self._ocr_capture_submitted = False
            self._captured_similarity = 0.0
            self._sharpness_threshold = 100.0
            self._empty_roi_frame: np.ndarray | None = None
            self._object_present_frames = 0
            self._object_absent_frames = 0
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
            self.video = RoiVideoLabel(roi)
            self.video.roi_changed.connect(self._roi_changed)
            self.video.setText("Starting...")
            self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.video.setMinimumSize(420, 260)
            self.video.setStyleSheet("background: #090b0e; color: #8d98a8;")
            self._inspection_overlay = QLabel(self)
            self._inspection_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._inspection_overlay.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            )
            self._inspection_overlay.hide()
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
            self.roi_button = QPushButton("Draw ROI")
            self.roi_button.setCheckable(True)
            self.roi_button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.roi_button.setStyleSheet(self.save_button.styleSheet())
            self.roi_button.toggled.connect(self.video.set_drawing)
            self.save_roi_button = QPushButton("Save ROI")
            self.save_roi_button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.save_roi_button.setStyleSheet(self.save_button.styleSheet())
            self.save_roi_button.clicked.connect(self.save_roi_requested)
            self.set_background_button = QPushButton("Set as Background")
            self.set_background_button.setCursor(
                Qt.CursorShape.PointingHandCursor
            )
            self.set_background_button.setStyleSheet(
                self.save_button.styleSheet()
            )
            self.set_background_button.clicked.connect(self.set_as_background)
            self.detect_button = QPushButton("Detect")
            self.detect_button.setEnabled(False)
            self.detect_button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.detect_button.setStyleSheet(
                "QPushButton {"
                " background: #16875b; color: white; border: none;"
                " border-radius: 5px; padding: 12px;"
                " font-size: 18px; font-weight: 800;"
                "}"
                "QPushButton:hover { background: #20a870; }"
                "QPushButton:disabled {"
                " background: #343b46; color: #7b8491;"
                "}"
            )
            self.detect_button.clicked.connect(self.request_detection)
            self.train_button = QPushButton("Train Model")
            self.train_button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.train_button.setStyleSheet(self.save_button.styleSheet())
            self.train_button.clicked.connect(self.train_model)
            self.train_button.setVisible(not self._automatic_detection)
            self.training_progress = QProgressBar()
            self.training_progress.setRange(0, 100)
            self.training_progress.setValue(0)
            self.training_progress.setTextVisible(True)
            self.training_progress.setVisible(False)
            self.match_badge = QLabel("PLACE THE OBJECT")
            self.match_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.match_badge.setStyleSheet(
                "color: #f4b942; font-size: 15px; font-weight: 700;"
            )
            layout.addWidget(heading)
            layout.addWidget(self.video, 1)
            layout.addWidget(self.match_badge)
            layout.addWidget(self.status)
            layout.addWidget(self.save_button)
            roi_layout = QHBoxLayout()
            roi_layout.addWidget(self.roi_button)
            roi_layout.addWidget(self.save_roi_button)
            roi_layout.addWidget(self.set_background_button)
            layout.addLayout(roi_layout)
            layout.addWidget(self.detect_button)
            layout.addWidget(self.train_button)
            layout.addWidget(self.training_progress)
            self.set_background_button.setVisible(not self._automatic_detection)
            self.detect_button.setVisible(not self._automatic_detection)
            self._load_model()
            self.show_message("Place the object")

        def set_repository_root(self, repository_root: Path) -> None:
            self._snapshot_dir = repository_root / self._camera_name
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)
            self._model_feature = None
            self._object_references = []
            self.prepare_for_camera()
            self._load_model()

        def resizeEvent(self, event: Any) -> None:
            super().resizeEvent(event)
            self._inspection_overlay.setGeometry(self.rect())

        def show_inspection_overlay(self, matched: bool) -> None:
            self._inspection_overlay.setText("✓" if matched else "✕")
            symbol_size = max(
                180,
                round(min(self.width(), self.height()) * 0.72),
            )
            self._inspection_overlay.setStyleSheet(
                "background: rgba(15, 18, 23, 145);"
                f"color: {'#35d07f' if matched else '#ff5f65'};"
                f"font-size: {symbol_size}px; font-weight: 900;"
            )
            self._inspection_overlay.setGeometry(self.rect())
            self._inspection_overlay.show()
            self._inspection_overlay.raise_()

        def clear_inspection_overlay(self) -> None:
            self._inspection_overlay.hide()
            self.detect_button.setEnabled(False)

        @property
        def roi(self) -> list[float] | None:
            return self.video.roi

        def set_roi(self, roi: list[float] | None) -> None:
            self.video.set_roi(roi)
            self.prepare_for_camera()

        def _roi_changed(self, _roi: object) -> None:
            self.prepare_for_camera()

        def prepare_for_camera(self) -> None:
            """Reset the automatic inspection when the camera or ROI changes."""
            self._placement_started = None
            self._presence_text = ""
            self._presence_readings = 0
            self._presence_started = None
            self._placement_missing_readings = 0
            self._window_best_score = 0.0
            self._window_best_metadata = {}
            self._window_has_confident_text = False
            self._last_result = None
            self._removal_readings = 0
            self._removal_started = None
            self._candidate_reference = ""
            self._pending_reference_frame = None
            self._active_reference_capture = False
            self._confident_match_readings = 0
            self._inspection_locked = False
            self._awaiting_ocr = False
            self._ocr_capture_submitted = False
            self._captured_similarity = 0.0
            self._latest_metadata = {}
            self._latest_inspection_frame = None
            self._empty_roi_frame = None
            self._object_present_frames = 0
            self._object_absent_frames = 0
            self._inspection_overlay.hide()
            if self._automatic_detection:
                self.match_badge.setText("AUTOMATIC DETECTION")
                self.show_message("Reading text automatically" if self._is_ocr else "Detecting objects automatically")
                self.inspection_reset.emit(self._camera_name)
                return
            self.match_badge.setText("SET EMPTY BACKGROUND")
            self.match_badge.setStyleSheet(
                "color: #f4b942; font-size: 15px; font-weight: 700;"
            )
            self.show_message("Keep ROI empty, then click Set as Background")
            self.inspection_reset.emit(self._camera_name)

        def set_as_background(self) -> None:
            """Use the currently visible clean ROI as the removal reference."""
            if self._latest_inspection_frame is None:
                self.show_message(
                    "No camera frame available for background capture",
                    error=True,
                )
                return
            roi_frame = crop_to_roi(self._latest_inspection_frame, self.roi)
            if roi_frame.size == 0:
                self.show_message("The selected ROI is invalid", error=True)
                return
            self._empty_roi_frame = cv2.cvtColor(
                roi_frame,
                cv2.COLOR_BGR2GRAY,
            ).copy()
            self._inspection_locked = False
            self._awaiting_ocr = False
            self._ocr_capture_submitted = False
            self._captured_similarity = 0.0
            self._object_present_frames = 0
            self._object_absent_frames = 0
            self._latest_metadata = {}
            self.clear_inspection_overlay()
            self.detect_button.setEnabled(True)
            self.match_badge.setText("PLACE THE OBJECT")
            self.match_badge.setStyleSheet(
                "color: #f4b942; font-size: 15px; font-weight: 700;"
            )
            self.show_message("Background saved. Place the object")
            self.inspection_reset.emit(self._camera_name)

        def request_detection(self) -> None:
            """Start one inspection only when requested by the operator."""
            if self._automatic_detection:
                if self._awaiting_ocr or self._latest_inspection_frame is None:
                    return
                self._active_reference_capture = self._pending_reference_frame is not None
                self._ocr_frame = (
                    self._pending_reference_frame
                    if self._active_reference_capture
                    else crop_to_roi(self._latest_inspection_frame, self.roi).copy()
                )
                self._pending_reference_frame = None
                self._awaiting_ocr = True
                self._ocr_capture_submitted = False
                return
            if self._empty_roi_frame is None:
                self.show_message(
                    "Set the empty background before detection",
                    error=True,
                )
                return
            if self._inspection_locked:
                self.show_message(
                    "Remove the current object before the next detection",
                    error=True,
                )
                return
            if self._awaiting_ocr:
                return
            if self._latest_inspection_frame is None:
                self.show_message("No camera frame available", error=True)
                return

            frame = self._latest_inspection_frame
            roi_frame = crop_to_roi(frame, self.roi)
            object_present, _current_roi = roi_object_difference(
                roi_frame,
                self._empty_roi_frame,
            )
            if not object_present:
                self.show_message(
                    "No object detected. Place the object and press Detect",
                    error=True,
                )
                return
            if image_sharpness(roi_frame) < self._sharpness_threshold:
                self.show_message(
                    "Image is not clear. Hold the object steady and press Detect",
                    error=True,
                )
                return

            similarity = 1.0
            if self._model_feature is not None:
                similarity = visual_similarity(
                    extract_visual_feature(frame),
                    self._model_feature,
                )
            self._captured_similarity = similarity
            self._awaiting_ocr = True
            self._ocr_capture_submitted = False
            self.detect_button.setEnabled(False)
            self.show_message(
                "Detecting and reading place text..."
                if self._is_ocr
                else "Checking the object against the trained model..."
            )

        def save_roi_requested(self) -> None:
            self.window().save_rois()

        @property
        def _model_path(self) -> Path:
            return self._snapshot_dir / "visual_match_model.npz"

        def _load_model(self) -> None:
            if not self._is_ocr and self._automatic_detection:
                try:
                    data = json.loads(self._reference_path.read_text(encoding="utf-8"))
                    self._object_references = [r for r in data["objects"]
                        if isinstance(r, dict) and isinstance(r.get("class_name"), str)
                        and len(r.get("feature", [])) == 768
                        and np.isfinite(np.asarray(r["feature"], dtype=np.float32)).all()]
                except (OSError, ValueError, KeyError, TypeError):
                    self._object_references = []
                self.match_badge.setText("WATCHING")
                return
            if self._automatic_detection:
                try:
                    data = json.loads(self._reference_path.read_text(encoding="utf-8"))
                    self._text_references = [str(t) for t in data["texts"] if isinstance(t, str) and t.strip()]
                except (OSError, ValueError, KeyError, TypeError):
                    self._text_references = []
                self.match_badge.setText("WATCHING")
                return
            try:
                with np.load(self._model_path) as model:
                    self._model_feature = model["feature"].astype(np.float32)
                    self._match_threshold = float(model["threshold"])
                self.match_badge.setText("PLACE THE OBJECT")
                self.match_badge.setStyleSheet(
                    "color: #f4b942; font-size: 15px; font-weight: 700;"
                )
            except (OSError, KeyError, ValueError):
                self._model_feature = None
                self.match_badge.setText("MODEL NOT TRAINED")
                self.match_badge.setStyleSheet(
                    "color: #f4b942; font-size: 15px; font-weight: 700;"
                )

        @property
        def _reference_path(self) -> Path:
            return self._snapshot_dir / ("text_references.json" if self._is_ocr else "object_references.json")

        @staticmethod
        def _normalize_text(text: str) -> str:
            return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())

        def _save_text_reference(self, text: str, confidence: float) -> None:
            self._active_reference_capture = False
            self.save_button.setEnabled(True)
            if not self._normalize_text(text) or confidence < self._ocr_confidence_threshold:
                self.show_message("Reference not saved: text unclear. Adjust ROI/lighting and click Save again", error=True)
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            snapshot_path = self._snapshot_dir / f"{self._camera_name}_{timestamp}.jpg"
            references = list(self._text_references)
            if text not in references:
                references.append(text)
            try:
                snapshot_path.write_bytes(encode_frame(self._ocr_frame))
                temporary = self._reference_path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"texts": references}, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(self._reference_path)
            except OSError as error:
                self.show_message(f"Reference save failed: {error}", error=True)
                return
            self._text_references = references
            self._confident_match_readings = 0
            self._candidate_reference = ""
            self._placement_started = None
            self._presence_text = ""
            self._presence_readings = 0
            self._presence_started = None
            self._placement_missing_readings = 0
            self._window_best_score = 0.0
            self._window_best_metadata = {}
            self._window_has_confident_text = False
            self._last_result = None
            self.clear_inspection_overlay()
            self.inspection_reset.emit(self._camera_name)
            self.match_badge.setText("WATCHING")
            self.show_message(f"Saved reference text: {text}")

        def _save_object_reference(self, objects: list[dict[str, Any]]) -> None:
            self._active_reference_capture = False
            self.save_button.setEnabled(True)
            if len(objects) != 1:
                self.show_message("Reference not saved: place one detectable object in the ROI and click Save", error=True)
                return
            reference = objects[0]
            references = self._object_references + [{
                "class_name": reference["class_name"], "feature": reference["feature"],
            }]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            try:
                (self._snapshot_dir / f"{self._camera_name}_{timestamp}.jpg").write_bytes(encode_frame(self._ocr_frame))
                temporary = self._reference_path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"objects": references}, indent=2), encoding="utf-8")
                temporary.replace(self._reference_path)
            except OSError as error:
                self.show_message(f"Reference save failed: {error}", error=True)
                return
            self._object_references = references
            self.prepare_for_camera()
            self.show_message(f"Saved object reference: {reference['class_name']}")

        def show_message(self, text: str, *, error: bool = False) -> None:
            self.status.setText(text)
            self.status.setStyleSheet(
                "color: #ff6b6b;" if error else "color: #8d98a8;"
            )

        @property
        def ocr_capture_submitted(self) -> bool:
            return self._ocr_capture_submitted

        def mark_ocr_capture_submitted(self) -> None:
            self._ocr_capture_submitted = True

        def accept_analysis(self, metadata: dict[str, Any]) -> None:
            self._latest_metadata = metadata
            if not self._awaiting_ocr:
                return
            self._awaiting_ocr = False
            ocr = metadata.get("ocr", {})
            text = str(ocr.get("text", "")).strip() if isinstance(ocr, dict) else ""
            if self._automatic_detection:
                if self._is_ocr:
                    words = ocr.get("words", []) if isinstance(ocr, dict) else []
                    confidence = min((float(w.get("confidence", 0)) for w in words if str(w.get("text", "")).strip()), default=0.0)
                    has_references = bool(self._text_references)
                else:
                    objects = [obj for obj in metadata.get("objects", [])
                               if float(obj.get("confidence", 0)) >= self._ocr_confidence_threshold]
                    text = "|".join(sorted(str(obj["class_name"]) for obj in objects))
                    confidence = min((float(obj["confidence"]) for obj in objects), default=0.0)
                    has_references = bool(self._object_references)
                if self._active_reference_capture:
                    self._ocr_capture_submitted = False
                    if self._is_ocr:
                        self._save_text_reference(text, confidence)
                    else:
                        self._save_object_reference(objects)
                    return
                if self._last_result is not None:
                    # Keep the accepted result through weak or different OCR readings.
                    # Rearm only after sustained absence of all readable text.
                    self._ocr_capture_submitted = False
                    if self._normalize_text(text):
                        self._removal_readings = 0
                        self._removal_started = None
                    else:
                        now = time.monotonic()
                        if self._removal_started is None:
                            self._removal_started = now
                        self._removal_readings += 1
                        if (
                            self._removal_readings >= self._required_removal_readings
                            and now - self._removal_started >= self._removal_hold_seconds
                        ):
                            self._placement_started = None
                            self._presence_text = ""
                            self._presence_readings = 0
                            self._presence_started = None
                            self._placement_missing_readings = 0
                            self._window_best_score = 0.0
                            self._window_best_metadata = {}
                            self._window_has_confident_text = False
                            self._last_result = None
                            self._confident_match_readings = 0
                            self._candidate_reference = ""
                            self._removal_readings = 0
                            self._removal_started = None
                            self.clear_inspection_overlay()
                            self.match_badge.setText("WATCHING")
                            self.match_badge.setStyleSheet(
                                "color: #f4b942; font-size: 24px; font-weight: 700;"
                            )
                            self.show_message("Place the next object")
                            self.inspection_reset.emit(self._camera_name)
                            return
                    self.show_message("Matched. Remove the object before the next inspection" if self._last_result else "Not matched. Remove the object before the next inspection")
                    return
                normalized = self._normalize_text(text)
                if self._is_ocr:
                    scores = [(SequenceMatcher(None, normalized, self._normalize_text(ref)).ratio(), ref)
                              for ref in self._text_references] if normalized else []
                else:
                    # One inspection object per ROI. Extra objects cannot produce a pass.
                    scores = [(visual_similarity(np.asarray(objects[0]["feature"], dtype=np.float32),
                                                 np.asarray(ref["feature"], dtype=np.float32)), ref["class_name"])
                              for ref in self._object_references
                              if len(objects) == 1 and objects[0]["class_name"] == ref["class_name"]]
                score, reference = max(scores, default=(0.0, ""))
                self._captured_similarity = score
                if reference != self._candidate_reference:
                    self._confident_match_readings = 0
                self._candidate_reference = reference
                matched = bool(reference) and confidence >= self._ocr_confidence_threshold and score >= self._automatic_match_threshold
            else:
                matched = (
                self._model_feature is not None
                and self._captured_similarity >= self._match_threshold
            )
            self._inspection_locked = not self._automatic_detection
            self._ocr_capture_submitted = False
            self._best_ocr_text = text
            if self._automatic_detection:
                now = time.monotonic()
                if self._placement_started is None:
                    reliable = (
                        len(normalized) >= self._placement_min_text_length
                        and confidence >= self._ocr_confidence_threshold
                    )
                    if not reliable:
                        self._presence_text = ""
                        self._presence_readings = 0
                        self._presence_started = None
                    elif (
                        self._presence_started is None
                        or SequenceMatcher(None, normalized, self._presence_text).ratio() < 0.8
                    ):
                        self._presence_text = normalized
                        self._presence_readings = 1
                        self._presence_started = now
                    else:
                        self._presence_readings += 1
                    if (
                        self._presence_readings < self._placement_confirm_readings
                        or self._presence_started is None
                        or now - self._presence_started < self._placement_confirm_seconds
                    ):
                        self.match_badge.setText("WATCHING")
                        self.show_message("Waiting for stable text in the ROI" if self._is_ocr else "Waiting for a stable object in the ROI")
                        return
                    self._placement_started = now
                    self._window_best_score = 0.0
                    self._window_best_metadata = metadata
                    self._window_has_confident_text = False
                # Do not issue a failure for a candidate that vanished during observation.
                if len(normalized) < self._placement_min_text_length or confidence < self._ocr_confidence_threshold:
                    self._placement_missing_readings += 1
                    if self._placement_missing_readings >= 3:
                        self._placement_started = None
                        self._presence_text = ""
                        self._presence_readings = 0
                        self._presence_started = None
                        self._window_best_score = 0.0
                        self._window_has_confident_text = False
                        self.match_badge.setText("WATCHING")
                        self.show_message("Waiting for stable text in the ROI" if self._is_ocr else "Waiting for a stable object in the ROI")
                    # Never decide from an empty or unreliable current reading.
                    return
                self._placement_missing_readings = 0
                if normalized and confidence >= self._ocr_confidence_threshold:
                    if not self._window_has_confident_text or score > self._window_best_score:
                        self._window_best_score = score
                        self._window_best_metadata = metadata
                    self._window_has_confident_text = True
                elapsed = now - self._placement_started
                if elapsed < self._placement_wait_seconds:
                    self.match_badge.setText("OBSERVING")
                    self.show_message(
                        f"Observing {elapsed:.1f}/{self._placement_wait_seconds:.1f}s"
                        f" | best score {self._window_best_score:.0%}"
                        f" | required {self._automatic_match_threshold:.0%}"
                    )
                    return
                matched = (
                    self._window_has_confident_text
                    and has_references
                    and self._window_best_score >= self._automatic_match_threshold
                )
                self._captured_similarity = self._window_best_score
                metadata = self._window_best_metadata
                self._latest_metadata = metadata
            self.match_badge.setText("CORRECT" if matched else "NOT MATCHED")
            self.match_badge.setStyleSheet(
                "color: #35d07f; font-size: 42px; font-weight: 900;"
                if matched
                else "color: #ff5f65; font-size: 24px; font-weight: 900;"
            )
            self.show_message("Matched" if matched else "Not matched", error=not matched)
            if self._automatic_detection:
                self.show_inspection_overlay(matched)
                self._last_result = matched
                self._removal_readings = 0
                self._removal_started = None
                self.show_message("Matched. Remove the object before the next inspection" if self._last_result else "Not matched. Remove the object before the next inspection")
                self.classification_changed.emit(self._camera_name, matched)
            self.inspection_completed.emit(
                self._camera_name,
                matched,
                self._captured_similarity,
                str(self._reference_path if self._automatic_detection else self._model_path),
                metadata,
            )
            if not self._automatic_detection:
                self.classification_changed.emit(self._camera_name, matched)

        def reset_inspection(self) -> None:
            if not self._inspection_locked and not self._awaiting_ocr:
                return
            self._inspection_locked = False
            self._awaiting_ocr = False
            self._ocr_capture_submitted = False
            self._captured_similarity = 0.0
            self._best_ocr_text = ""
            self._latest_metadata = {}
            self._object_present_frames = 0
            self._object_absent_frames = 0
            self.match_badge.setText("PLACE THE OBJECT")
            self.match_badge.setStyleSheet(
                "color: #f4b942; font-size: 15px; font-weight: 700;"
            )
            self.show_message("Place the object")
            self.detect_button.setEnabled(True)
            self.window().clear_inspection_overlay(self._camera_name)
            self.inspection_reset.emit(self._camera_name)

        def show_jpeg(
            self,
            data: bytes,
            metadata: dict[str, Any] | None = None,
            inspection_frame: np.ndarray | None = None,
        ) -> None:
            image = QImage.fromData(data, "JPG")
            if image.isNull():
                self.show_message("Invalid frame received", error=True)
                return
            self._latest_jpeg = data
            self._latest_inspection_frame = (
                inspection_frame.copy()
                if inspection_frame is not None
                else decode_frame(data)
            )
            if metadata is not None:
                self._latest_metadata = metadata
            self.save_button.setEnabled(self._pending_reference_frame is None and not self._active_reference_capture)
            self.video.setPixmap(
                QPixmap.fromImage(image).scaled(
                    self.video.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self._display_frame_count += 1
            fps_elapsed = time.monotonic() - self._display_fps_started
            if fps_elapsed >= 1.0:
                self._display_fps = self._display_frame_count / fps_elapsed
                self._display_frame_count = 0
                self._display_fps_started = time.monotonic()
            if self._automatic_detection:
                self.request_detection()
                return
            ocr = self._latest_metadata.get("ocr", {})
            ocr_status = ""
            if isinstance(ocr, dict) and ocr.get("inference_time_ms") is not None:
                ocr_status = f" | OCR {float(ocr['inference_time_ms']):.0f} ms async"
            if (
                self._empty_roi_frame is not None
                and not self._inspection_locked
                and not self._awaiting_ocr
                and self._object_present_frames == 0
            ):
                self.status.setText("Place the object")
            self._match_frame_counter += 1
            if self._match_frame_counter % 5 == 0:
                frame = (
                    self._latest_inspection_frame
                    if self._latest_inspection_frame is not None
                    else decode_frame(data)
                )
                roi_frame = crop_to_roi(frame, self.roi)

                if self._empty_roi_frame is None:
                    self.match_badge.setText("SET EMPTY BACKGROUND")
                    self.show_message(
                        "Keep ROI empty, then click Set as Background"
                    )
                    return

                object_present, current_roi = roi_object_difference(
                    roi_frame,
                    self._empty_roi_frame,
                )

                if self._inspection_locked:
                    if object_present:
                        self._object_absent_frames = 0
                    else:
                        self._object_absent_frames += 1
                        if self._object_absent_frames >= 3:
                            self.reset_inspection()
                    return

                if object_present:
                    self._object_present_frames += 1
                    self._object_absent_frames = 0
                    if not self._inspection_locked and not self._awaiting_ocr:
                        self.show_message("Object ready. Press Detect")
                    self._previously_matched = True
                else:
                    self._object_present_frames = 0
                    self._object_absent_frames += 1
                    if self._awaiting_ocr and self._object_absent_frames >= 3:
                        self.reset_inspection()
                    elif not self._awaiting_ocr:
                        self._empty_roi_frame = cv2.addWeighted(
                            self._empty_roi_frame,
                            0.9,
                            current_roi,
                            0.1,
                            0.0,
                        )
                    self._previously_matched = False
                    self._ocr_collection_started = None
                    self._best_ocr_metadata = {}
                    self._best_ocr_text = ""

        def save_snapshot(self) -> None:
            if self._automatic_detection:
                if self._latest_inspection_frame is None:
                    self.show_message("No frame available to save", error=True)
                    return
                self._pending_reference_frame = crop_to_roi(self._latest_inspection_frame, self.roi).copy()
                self.save_button.setEnabled(False)
                self.show_message("Reading captured reference text..." if self._is_ocr else "Reading captured object reference...")
                return
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
            self._audio_alert_manager = AudioAlertManager(
                AudioAlertConfigStore(CONFIG_PATH)
            )
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
            self._camera_rois = load_camera_rois()
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
            audio_alert_button = QPushButton("Audio Alert Config")
            audio_alert_button.setStyleSheet(
                select_repository_button.styleSheet()
            )
            audio_alert_button.clicked.connect(
                self._open_audio_alert_config
            )
            repository_layout.addWidget(repository_title)
            repository_layout.addWidget(self._repository_path_label, 1)
            repository_layout.addWidget(select_repository_button)
            repository_layout.addWidget(camera_settings_button)
            repository_layout.addWidget(transactions_button)
            repository_layout.addWidget(audio_alert_button)
            layout.addWidget(repository_bar, 0, 0, 1, 3)

            specifications = (
                ("Embossed Text Camera", "ocr", args.ocr_device),
                ("Object Detection Camera 1", "object", args.object_device),
                ("Object Detection Camera 2", "object", args.object_device),
            )
            self._panels: list[CameraPanel] = []
            self._input_queues: list[Any] = []
            self._output_queues: list[Any] = []
            self._workers: list[Any] = []
            self._latest_analysis_metadata: list[dict[str, Any]] = [
                {},
                {},
                {},
            ]

            for column, (title, kind, device) in enumerate(specifications):
                camera_name = f"camera-{column + 1}"
                panel = CameraPanel(
                    title,
                    camera_name,
                    self._repository_root,
                    self._match_event_cooldown,
                    self._camera_rois.get(camera_name),
                )
                panel.inspection_completed.connect(self._record_camera_inspection)
                panel.inspection_completed.connect(self._show_inspection_overlay)
                panel.inspection_reset.connect(self._clear_analysis_metadata)
                panel.inspection_reset.connect(self._audio_alert_manager.reset_classification)
                panel.classification_changed.connect(
                    self._audio_alert_manager.handle_classification
                )
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

        def _show_inspection_overlay(
            self,
            camera_name: str,
            matched: bool,
            _similarity: float,
            _model_path: str,
            _metadata: object,
        ) -> None:
            panel_index = int(camera_name.rsplit("-", 1)[-1]) - 1
            if 0 <= panel_index < len(self._panels):
                self._panels[panel_index].show_inspection_overlay(matched)

        def clear_inspection_overlay(self, camera_name: str) -> None:
            panel_index = int(camera_name.rsplit("-", 1)[-1]) - 1
            if 0 <= panel_index < len(self._panels):
                self._panels[panel_index].clear_inspection_overlay()

        def _clear_analysis_metadata(self, camera_name: str) -> None:
            panel_index = int(camera_name.rsplit("-", 1)[-1]) - 1
            if 0 <= panel_index < len(self._latest_analysis_metadata):
                self._latest_analysis_metadata[panel_index] = {}

        def save_rois(self) -> None:
            rois = {
                f"camera-{index + 1}": panel.roi
                for index, panel in enumerate(self._panels)
            }
            try:
                save_camera_rois(rois)
            except OSError as error:
                QMessageBox.critical(self, "ROI Settings", f"Could not save ROI settings: {error}")
                return
            self._camera_rois = rois
            for panel in self._panels:
                panel.roi_button.setChecked(False)
            self._panels[0].show_message("ROI saved; OCR is using the selected area")

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

        def _record_camera_inspection(
            self,
            camera_name: str,
            matched: bool,
            similarity: float,
            model_path: str,
            metadata: object,
        ) -> None:
            """Save one completed OCR or object inspection result."""
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
                "matched": matched,
                "similarity": round(similarity, 4),
                "model_path": model_path,
            }
            if captured_text:
                captured_data["ocr_text"] = captured_text
            if isinstance(metadata, dict) and "objects" in metadata:
                captured_data["objects"] = [
                    {key: value for key, value in obj.items() if key != "feature"}
                    for obj in metadata["objects"]
                ]
            inspection_kind = "place_text_match" if camera_name == "camera-1" else "object_match"
            send_result = (
                self._inspection_sender.send_pass
                if matched
                else self._inspection_sender.send_fail
            )
            result = send_result(
                camera_id=camera_name,
                section_id=camera_name.upper(),
                event_type=inspection_kind,
                captured_data=captured_data,
                confidence=round(similarity, 4),
                evidence_link=model_path,
                comments=(
                    f"{camera_name} {inspection_kind} matched"
                    if matched
                    else f"{camera_name} {inspection_kind} did not match"
                ),
                remarks="Automatically generated by Gibraltar Visual AI",
            )
            panel_index = int(camera_name.rsplit("-", 1)[-1]) - 1
            if 0 <= panel_index < len(self._panels):
                if result.get("success"):
                    self._panels[panel_index].show_message(
                        f"{'Matched' if matched else 'Not matched'}; event saved ({result['event_id']})"
                    )
                else:
                    self._panels[panel_index].show_message(
                        "Match queued; waiting for database connection",
                        error=True,
                    )

        def _open_audio_alert_config(self) -> None:
            dialog = AudioAlertConfigDialog(
                self._audio_alert_manager.settings,
                self,
            )
            if dialog.exec() == QDialog.DialogCode.Accepted:
                try:
                    self._audio_alert_manager.update(dialog.settings())
                except OSError as error:
                    QMessageBox.critical(
                        self,
                        "Audio Alert Config",
                        f"Could not save audio settings: {error}",
                    )
                    return
                QMessageBox.information(
                    self,
                    "Audio Alert Config",
                    "Audio alert configuration saved.",
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
                panel.prepare_for_camera()
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
                    # Camera 1 runs OCR. This camera's YUY2 USB mode delivers
                    # ~7 FPS at 720p but near 30 FPS at 640x480, and the lower
                    # resolution also materially reduces OCR inference time.
                    usb_width = 640 if index == 0 else args.width
                    usb_height = 480 if index == 0 else args.height
                    camera = LocalCamera(
                        int(source_value),
                        width=usb_width,
                        height=usb_height,
                        fps=args.fps,
                    )
                    source_label = (
                        f"USB {source_value} | {usb_width}x{usb_height}"
                    )
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

            for camera_index, (camera, panel, input_queue) in enumerate(zip(
                self._cameras,
                self._panels,
                self._input_queues,
            )):
                if camera is None:
                    continue
                ok, frame = camera.read_latest()
                if ok:
                    if camera_index == 0 and max(frame.shape[:2]) > 960:
                        scale = 960.0 / max(frame.shape[:2])
                        frame = cv2.resize(
                            frame,
                            None,
                            fx=scale,
                            fy=scale,
                            interpolation=cv2.INTER_AREA,
                        )
                    draw_overlay = draw_ocr_overlay if camera_index == 0 else draw_object_overlay
                    display_frame = draw_overlay(frame, self._latest_analysis_metadata[camera_index], panel.roi)
                    panel.show_jpeg(encode_frame(display_frame), self._latest_analysis_metadata[camera_index], inspection_frame=frame)
                    if panel.ocr_capture_submitted or not panel._awaiting_ocr:
                        continue
                    encoded = encode_frame(panel._ocr_frame)
                    panel.mark_ocr_capture_submitted()
                    latest_put(input_queue, encoded)

            for panel_index, (panel, output_queue, worker) in enumerate(zip(
                self._panels,
                self._output_queues,
                self._workers,
            )):
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
                    elif message_type == "analysis":
                        if isinstance(payload, dict):
                            self._latest_analysis_metadata[panel_index] = payload
                            panel.accept_analysis(payload)
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
