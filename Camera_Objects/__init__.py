"""Camera-specific object and event monitors."""

from .bathroom import BathroomEntryMonitor
from .flame import FlameMonitor
from .people_presence import PeoplePresenceMonitor
from .mask_detection import MaskDetectionMonitor
from .smoke_detection import SmokeDetectionMonitor
from .gloves_detection import GlovesDetectionMonitor
from .gun_detection import GunDetectionMonitor
from .cycle_detection import CycleDetectionMonitor

__all__ = [
    "BathroomEntryMonitor",
    "CycleDetectionMonitor",
    "FlameMonitor",
    "GlovesDetectionMonitor",
    "GunDetectionMonitor",
    "MaskDetectionMonitor",
    "PeoplePresenceMonitor",
    "SmokeDetectionMonitor",
]
