"""Windows master-output volume adapter."""

from __future__ import annotations


class SystemVolumeController:
    """Read and update the Windows default speaker volume as 0..100 percent."""

    def __init__(self) -> None:
        self._endpoint = None
        self.error = ""
        try:
            from pycaw.pycaw import AudioUtilities

            self._endpoint = AudioUtilities.GetSpeakers().EndpointVolume
        except Exception as error:  # Core Audio availability is machine-specific.
            self.error = str(error)

    @property
    def available(self) -> bool:
        return self._endpoint is not None

    def get_percent(self) -> int:
        if self._endpoint is None:
            raise RuntimeError(self.error or "System volume is unavailable")
        scalar = float(self._endpoint.GetMasterVolumeLevelScalar())
        return max(0, min(100, round(scalar * 100)))

    def set_percent(self, percent: int) -> None:
        if self._endpoint is None:
            raise RuntimeError(self.error or "System volume is unavailable")
        scalar = max(0, min(100, int(percent))) / 100.0
        self._endpoint.SetMasterVolumeLevelScalar(scalar, None)
