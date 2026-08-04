"""Gibraltar three-panel visual AI dashboard using one built-in camera.

The camera is opened once. Every captured frame is broadcast to one OCR worker
and two object-detection workers, then displayed in three PySide6 panels.
Repository-level OCR and detection modules are referenced rather than copied.
"""

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
REPOSITORY_DIR = PROJECT_DIR.parents[2]
CONFIG_PATH = PROJECT_DIR / "config.json"
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from camera import LocalCamera  # noqa: E402
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


def create_ocr_processor(device: str) -> Callable[[np.ndarray], np.ndarray]:
    from OCR.PaddleOCR import OCR, OCRConfig
    from OCR.PaddleOCR.drawing import draw_result

    reader = OCR(OCRConfig(device=device))

    def process(frame: np.ndarray) -> np.ndarray:
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
        return output

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
            output = processor(frame)
            latest_put(output_queue, ("frame", encode_frame(output)))
    except Exception as error:
        if not stop_event.is_set():
            latest_put(
                output_queue,
                ("error", f"{type(error).__name__}: {error}"),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gibraltar built-in-camera visual AI dashboard",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="built-in/OpenCV camera index (default: 0)",
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
        QFileDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QProgressBar,
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

    class CameraPanel(QFrame):
        match_detected = Signal(str, float, str)

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
            self._model_feature: np.ndarray | None = None
            self._match_threshold = 0.75
            self._match_frame_counter = 0
            self._match_event_cooldown = match_event_cooldown
            self._last_match_event_time = 0.0
            self._previously_matched = False
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

        def show_jpeg(self, data: bytes) -> None:
            image = QImage.fromData(data, "JPG")
            if image.isNull():
                self.show_message("Invalid frame received", error=True)
                return
            self._latest_jpeg = data
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
            self._camera = LocalCamera(
                args.camera,
                width=args.width,
                height=args.height,
                fps=args.fps,
            )
            try:
                self._camera.open()
            except ConnectionError:
                pass

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
            repository_layout.addWidget(repository_title)
            repository_layout.addWidget(self._repository_path_label, 1)
            repository_layout.addWidget(select_repository_button)
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
                panel.match_detected.connect(self._record_match)
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

            if not self._camera.is_opened:
                for panel in self._panels:
                    panel.video.setText("Camera unavailable")
                    panel.show_message(
                        f"Could not open built-in camera {args.camera}",
                        error=True,
                    )

            self._timer = QTimer(self)
            self._timer.timeout.connect(self._update)
            self._timer.start(max(1, round(1000 / args.fps)))

        def _record_match(
            self,
            camera_name: str,
            similarity: float,
            model_path: str,
        ) -> None:
            """Send one positive match to the existing inspection backend."""
            result = self._inspection_sender.send_pass(
                camera_id=camera_name,
                section_id=camera_name.upper(),
                event_type="trained_model_match",
                captured_data={
                    "matched": True,
                    "similarity": round(similarity, 4),
                    "database": self._database_name,
                    "table": self._database_table,
                    "model_path": model_path,
                },
                confidence=round(similarity, 4),
                evidence_link=model_path,
                comments="Live frame matched the trained camera model",
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
                        "Match detected; database backend is disconnected",
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

        def _update(self) -> None:
            if shutdown_requested.is_set():
                self.close()
                return

            if self._camera.is_opened:
                ok, frame = self._camera.read()
                if ok:
                    encoded = encode_frame(frame)
                    for input_queue in self._input_queues:
                        latest_put(input_queue, encoded)
                else:
                    for panel in self._panels:
                        panel.show_message("Camera frame unavailable", error=True)

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
                        panel.show_jpeg(payload)
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
            self._camera.release()
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
    window.show()
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
