"""PySide6 dialog and playback controller for persisted audio alerts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QSoundEffect
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .config import CAMERA_NAMES, EVENT_NAMES, AudioAlertConfigStore
from .system_volume import SystemVolumeController


DEFAULT_AUDIO_DIR = Path(__file__).resolve().parents[1] / "resources" / "audio"
DEFAULT_AUDIO_FILES = {
    "matched": DEFAULT_AUDIO_DIR / "default_matched.wav",
    "mismatch": DEFAULT_AUDIO_DIR / "default_mismatch.wav",
}


class AudioSourceRow(QWidget):
    test_requested = Signal(str, object)

    def __init__(
        self,
        title: str,
        event_name: str,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        self._event_name = event_name
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.enabled = QCheckBox(title)
        self.enabled.setChecked(bool(settings["enabled"]))
        self.kind = QComboBox()
        self.kind.addItems(("Tone", "Voice"))
        self.kind.setCurrentText(str(settings["kind"]).title())
        self.use_default = QCheckBox("Use default tone")
        self.use_default.setChecked(bool(settings["use_default"]))
        self.path = QLineEdit(str(settings["path"]))
        self.path.setPlaceholderText("Select an audio file")
        self.path.setReadOnly(True)
        self.browse = QPushButton("Browse...")
        self.browse.clicked.connect(self._browse)
        self.test_button = QPushButton("Test")
        self.test_button.clicked.connect(
            lambda: self.test_requested.emit(self._event_name, self.value())
        )
        self.use_default.toggled.connect(self._update_source_controls)
        layout.addWidget(self.enabled)
        source_group = QGroupBox("Audio")
        source_layout = QHBoxLayout(source_group)
        source_layout.addWidget(self.use_default)
        source_layout.addWidget(self.kind)
        source_layout.addWidget(self.path, 1)
        source_layout.addWidget(self.browse)
        source_layout.addWidget(self.test_button)
        layout.addWidget(source_group, 1)
        self._update_source_controls(self.use_default.isChecked())

    def _update_source_controls(self, use_default: bool) -> None:
        self.kind.setEnabled(not use_default)
        self.path.setEnabled(not use_default)
        self.browse.setEnabled(not use_default)
        if use_default:
            self.path.setPlaceholderText(
                DEFAULT_AUDIO_FILES[self._event_name].name
            )
        else:
            self.path.setPlaceholderText("Select an audio file")

    def _browse(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {self.kind.currentText().lower()} audio",
            self.path.text(),
            "Audio files (*.wav *.mp3 *.ogg *.m4a *.aac *.flac);;All files (*)",
        )
        if selected:
            self.path.setText(selected)

    def value(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled.isChecked(),
            "kind": self.kind.currentText().lower(),
            "use_default": self.use_default.isChecked(),
            "path": self.path.text().strip(),
        }


class AudioAlertConfigDialog(QDialog):
    def __init__(self, settings: dict[str, Any], parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Audio Alert Config")
        self.setModal(True)
        self.resize(850, 520)
        self._camera_controls: dict[str, dict[str, Any]] = {}
        self._system_volume = SystemVolumeController()
        configured_volume = int(settings.get("system_volume", 80))
        self._initial_system_volume = configured_volume
        if self._system_volume.available:
            self._initial_system_volume = self._system_volume.get_percent()
        self._preview_audio_output = QAudioOutput(self)
        self._preview_audio_output.setVolume(1.0)
        self._preview_player = QMediaPlayer(self)
        self._preview_player.setAudioOutput(self._preview_audio_output)
        layout = QVBoxLayout(self)
        heading = QLabel("Configure match and mismatch audio for each camera")
        heading.setStyleSheet("font-size: 17px; font-weight: 700;")
        layout.addWidget(heading)
        volume_group = QGroupBox("System Volume")
        volume_layout = QHBoxLayout(volume_group)
        self._volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setSingleStep(1)
        self._volume_slider.setPageStep(5)
        self._volume_slider.setValue(
            self._initial_system_volume
            if self._system_volume.available
            else configured_volume
        )
        self._volume_value = QLabel(f"{self._volume_slider.value()}%")
        self._volume_value.setMinimumWidth(44)
        self._volume_slider.valueChanged.connect(self._volume_changed)
        volume_layout.addWidget(QLabel("Master speaker volume:"))
        volume_layout.addWidget(self._volume_slider, 1)
        volume_layout.addWidget(self._volume_value)
        if not self._system_volume.available:
            self._volume_slider.setEnabled(False)
            self._volume_slider.setToolTip(
                self._system_volume.error or "Windows system volume unavailable"
            )
        layout.addWidget(volume_group)
        for index, camera_name in enumerate(CAMERA_NAMES, start=1):
            camera_settings = settings[camera_name]
            group = QGroupBox(f"Camera {index}")
            group_layout = QVBoxLayout(group)
            alarm_enabled = QCheckBox("Alarm enabled")
            alarm_enabled.setChecked(bool(camera_settings["enabled"]))
            matched = AudioSourceRow(
                "Matched alert", "matched", camera_settings["matched"]
            )
            mismatch = AudioSourceRow(
                "Mismatch alert", "mismatch", camera_settings["mismatch"]
            )
            matched.test_requested.connect(self._test_audio)
            mismatch.test_requested.connect(self._test_audio)
            group_layout.addWidget(alarm_enabled)
            group_layout.addWidget(matched)
            group_layout.addWidget(mismatch)
            layout.addWidget(group)
            self._camera_controls[camera_name] = {
                "enabled": alarm_enabled,
                "matched": matched,
                "mismatch": mismatch,
            }
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def settings(self) -> dict[str, Any]:
        settings: dict[str, Any] = {
            camera_name: {
                "enabled": controls["enabled"].isChecked(),
                "matched": controls["matched"].value(),
                "mismatch": controls["mismatch"].value(),
            }
            for camera_name, controls in self._camera_controls.items()
        }
        settings["system_volume"] = self._volume_slider.value()
        return settings

    def _volume_changed(self, value: int) -> None:
        self._volume_value.setText(f"{value}%")
        if self._system_volume.available:
            try:
                self._system_volume.set_percent(value)
            except RuntimeError:
                self._volume_slider.setEnabled(False)

    def reject(self) -> None:
        if self._system_volume.available:
            try:
                self._system_volume.set_percent(self._initial_system_volume)
            except RuntimeError:
                pass
        super().reject()

    def _validate_and_accept(self) -> None:
        missing: list[str] = []
        for index, camera_name in enumerate(CAMERA_NAMES, start=1):
            settings = self.settings()[camera_name]
            if not settings["enabled"]:
                continue
            for event_name in EVENT_NAMES:
                event = settings[event_name]
                if event["enabled"] and event["use_default"]:
                    if not DEFAULT_AUDIO_FILES[event_name].is_file():
                        missing.append(f"Camera {index} default {event_name}")
                elif event["enabled"] and not Path(event["path"]).is_file():
                    missing.append(f"Camera {index} {event_name}")
        if missing:
            QMessageBox.warning(
                self,
                "Audio File Required",
                "Select an existing audio file for: " + ", ".join(missing),
            )
            return
        self.accept()

    def _test_audio(self, event_name: str, settings: object) -> None:
        if not isinstance(settings, dict):
            return
        audio_path = (
            DEFAULT_AUDIO_FILES[event_name]
            if settings.get("use_default")
            else Path(str(settings.get("path", "")))
        )
        if not audio_path.is_file():
            QMessageBox.warning(
                self,
                "Audio Test",
                "Select an existing audio file before testing.",
            )
            return
        self._preview_player.stop()
        self._preview_player.setSource(
            QUrl.fromLocalFile(str(audio_path.resolve()))
        )
        self._preview_player.play()


class AudioAlertManager:
    """Play one alert per camera classification transition."""

    def __init__(self, store: AudioAlertConfigStore) -> None:
        self._store = store
        self._settings = store.load()
        self._last_state: dict[str, bool] = {}
        self._audio_output = QAudioOutput()
        self._audio_output.setVolume(1.0)
        self._player = QMediaPlayer()
        self._player.setAudioOutput(self._audio_output)
        self._default_effects: dict[str, QSoundEffect] = {}
        for event_name, audio_path in DEFAULT_AUDIO_FILES.items():
            effect = QSoundEffect()
            effect.setAudioDevice(self._audio_output.device())
            effect.setSource(QUrl.fromLocalFile(str(audio_path.resolve())))
            effect.setVolume(1.0)
            self._default_effects[event_name] = effect

    @property
    def settings(self) -> dict[str, Any]:
        return deepcopy(self._settings)

    def update(self, settings: object) -> None:
        self._settings = self._store.save(settings)

    def reset_classification(self, camera_name: str) -> None:
        """Rearm match audio without playing a mismatch alert."""
        self._last_state.pop(camera_name, None)

    def handle_classification(self, camera_name: str, matched: bool) -> bool:
        previous = self._last_state.get(camera_name)
        self._last_state[camera_name] = matched
        if previous is not None and previous == matched:
            return False
        camera = self._settings.get(camera_name, {})
        event = camera.get("matched" if matched else "mismatch", {})
        if not camera.get("enabled") or not event.get("enabled"):
            return False
        event_name = "matched" if matched else "mismatch"
        audio_path = (
            DEFAULT_AUDIO_FILES[event_name]
            if event.get("use_default")
            else Path(str(event.get("path", "")))
        )
        if not audio_path.is_file():
            return False
        if event.get("use_default"):
            self._default_effects[event_name].play()
        else:
            self._player.stop()
            self._player.setSource(QUrl.fromLocalFile(str(audio_path.resolve())))
            self._player.play()
        return True
