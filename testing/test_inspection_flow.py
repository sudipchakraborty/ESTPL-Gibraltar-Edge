"""Offscreen regression checks without cameras, model loading, or backend delivery.

Run: .venv/Scripts/python.exe -m unittest testing.test_inspection_flow -v
"""
import ast
import base64
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import cv2
from PySide6.QtWidgets import QApplication
from Projects import main as app_module


def panel_namespace():
    # GUI classes are local to run_gui. Load those definitions without its camera,
    # worker, socket, or startup-transaction side effects.
    tree = ast.parse(Path(app_module.__file__).read_text(encoding="utf-8"))
    gui = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_gui")
    nodes = [n for n in gui.body if isinstance(n, ast.ImportFrom)]
    nodes += [n for n in gui.body if isinstance(n, ast.ClassDef)
              and n.name in {"RoiVideoLabel", "TrainingSignals", "CameraPanel"}]
    namespace = vars(app_module).copy()
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "inspection-gui-definitions", "exec"), namespace)
    return namespace


class InspectionFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.ns = panel_namespace()
        self.clock = 0.0
        self.ns["time"] = SimpleNamespace(monotonic=lambda: self.clock)
        self.ns["load_project_config"] = lambda: {}
        self.panels = []
        self.frame = np.full((80, 100, 3), (0, 0, 200), dtype=np.uint8)
        self.feature = app_module.extract_visual_feature(self.frame).tolist()

    def panel(self, index):
        p = self.ns["CameraPanel"]("Test", f"camera-{index}", Path(self.folder.name), 30, None)
        self.panels.append(p)
        self.addCleanup(p.deleteLater)
        p.events = []
        p.alerts = []
        p.inspection_completed.connect(lambda *event: p.events.append(event))
        p.classification_changed.connect(lambda *event: p.alerts.append(event))
        return p

    def metadata(self, index, label="bottle", confidence=.95):
        if index == 1:
            return {"ocr": {"text": label, "words": [{"text": label, "confidence": confidence}]}}
        return {"objects": [] if not label else [{"class_name": label, "class_id": 39,
                "confidence": confidence, "bbox": [0, 0, 100, 80], "feature": self.feature}],
                "image_width": 100, "image_height": 80}

    def read(self, p, metadata):
        self.clock += .5
        p._awaiting_ocr = True
        p.accept_analysis(metadata)

    def save(self, p, index):
        p._latest_inspection_frame = self.frame.copy()
        p.save_snapshot()
        p._latest_inspection_frame[:] = 0
        p.request_detection()
        np.testing.assert_array_equal(p._ocr_frame, self.frame)
        self.read(p, self.metadata(index))
        self.assertTrue(p._reference_path.is_file())
        self.assertFalse(p.events)
        p._load_model()

    def test_camera_1_save_match_latch_remove_mismatch(self):
        for index in (1,):
            with self.subTest(camera=index):
                p = self.panel(index)
                self.save(p, index)
                self.assertTrue(p.detect_button.isHidden())
                self.assertTrue(p.set_background_button.isHidden())
                self.assertTrue(p.train_button.isHidden())
                for _ in range(6):
                    self.read(p, self.metadata(index))
                self.assertFalse(p.events, "No decision before confirmation plus observation")
                self.read(p, self.metadata(index))
                self.assertTrue(p.events[-1][1])
                for _ in range(12):
                    self.read(p, self.metadata(index))
                self.assertEqual(len(p.alerts), 1)
                for _ in range(5):
                    self.read(p, self.metadata(index, ""))
                self.assertIsNone(p._last_result)
                self.assertTrue(p._inspection_overlay.isHidden())
                for _ in range(7):
                    self.read(p, self.metadata(index, "chair"))
                self.assertFalse(p.events[-1][1])
                self.assertEqual(len(p.alerts), 2)
                self.assertEqual(p._inspection_overlay.text(), "✕")

    def test_glitches_and_vanished_candidates_do_not_fail(self):
        for index in (1,):
            p = self.panel(index)
            for _ in range(10):
                self.read(p, self.metadata(index))
                self.read(p, self.metadata(index, ""))
            for _ in range(8):
                self.read(p, self.metadata(index, confidence=.1))
            self.assertIsNone(p._placement_started)
            for _ in range(3):
                self.read(p, self.metadata(index))
            for _ in range(6):
                self.read(p, self.metadata(index, ""))
            self.assertIsNone(p._placement_started)
            self.assertFalse(p.events)

    def test_object_cameras_save_and_train_in_separate_folders(self):
        first, second = self.panel(2), self.panel(3)
        encoded, jpeg = cv2.imencode(".jpg", self.frame)
        self.assertTrue(encoded)
        for panel in (first, second):
            self.assertFalse(panel.save_button.isHidden())
            self.assertFalse(panel.train_button.isHidden())
            self.assertFalse(panel.detect_button.isHidden())
            self.assertFalse(panel.set_background_button.isHidden())
            panel._latest_jpeg = jpeg.tobytes()
            panel.save_snapshot()

        first_images = list((Path(self.folder.name) / "camera-2").glob("*.jpg"))
        second_images = list((Path(self.folder.name) / "camera-3").glob("*.jpg"))
        self.assertEqual(len(first_images), 1)
        self.assertEqual(len(second_images), 1)
        self.assertNotEqual(first_images[0].parent, second_images[0].parent)

        first._train_model_worker(first_images)
        first._load_model()
        self.assertTrue(first._model_path.is_file())
        self.assertIsNotNone(first._model_feature)
        self.assertFalse(second._model_path.exists())

    def test_object_camera_uses_visual_model_without_ocr_text(self):
        p = self.panel(2)
        p._model_feature = app_module.extract_visual_feature(self.frame)
        p._captured_similarity = 1.0
        p._awaiting_ocr = True
        p.accept_analysis(self.metadata(2))
        self.assertTrue(p.events[-1][1])

    def test_object_transaction_contains_detection_details(self):
        tree = ast.parse(Path(app_module.__file__).read_text(encoding="utf-8"))
        window = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "GibraltarWindow")
        method = next(n for n in window.body if isinstance(n, ast.FunctionDef) and n.name == "_record_camera_inspection")
        namespace = vars(app_module).copy()
        exec(compile(ast.Module(body=[method], type_ignores=[]), "transaction-test", "exec"), namespace)
        sender = SimpleNamespace(send_pass=Mock(return_value={"success": True, "event_id": "test"}),
                                 send_fail=Mock(return_value={"success": True, "event_id": "test"}))
        window_stub = SimpleNamespace(_inspection_sender=sender, _panels=[Mock(), Mock(), Mock()])
        window_stub._panels[2]._ocr_frame = self.frame.copy()
        window_stub._panels[2]._latest_jpeg = app_module.encode_frame(np.zeros_like(self.frame))
        namespace["_record_camera_inspection"](window_stub, "camera-3", False, .4, "object_references.json", self.metadata(3))
        payload = sender.send_fail.call_args.kwargs
        self.assertEqual(payload["event_type"], "object_match")
        self.assertEqual(payload["camera_id"], "camera-3")
        self.assertEqual(payload["captured_data"]["objects"][0]["class_name"], "bottle")
        self.assertNotIn("feature", payload["captured_data"]["objects"][0])
        self.assertEqual(payload["evidence_image"]["content_type"], "image/jpeg")
        evidence = base64.b64decode(payload["evidence_image"]["data"])
        self.assertEqual(evidence, app_module.encode_frame(self.frame))
        self.assertNotIn("evidence_link", payload)
        namespace["_record_camera_inspection"](window_stub, "camera-3", True, .99, "object_references.json", self.metadata(3))
        self.assertEqual(sender.send_pass.call_args.kwargs["evidence_image"], payload["evidence_image"])

    def test_detector_emits_features_from_unannotated_object_crop(self):
        detector = Mock()
        detector.detect.return_value = (self.metadata(2)["objects"], [])
        with patch.dict("sys.modules", {"models.detector": SimpleNamespace(ObjectDetector=lambda *a, **k: detector)}):
            process = app_module.create_detection_processor("cpu")
            result = process(self.frame)
        np.testing.assert_allclose(result["objects"][0]["feature"], self.feature)
        self.assertEqual(result["image_width"], 100)
        self.assertFalse(detector.draw.called)


if __name__ == "__main__":
    unittest.main()
