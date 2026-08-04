"""
Display Utility Functions
Author : DeepVision Industrial Analytics
"""

from typing import List


def print_detections(detections: List[dict]) -> None:
    """
    Print all detected objects in a formatted table.
    """

    if not detections:
        print("\nNo objects detected.\n")
        return

    print("\n" + "=" * 90)
    print(f"{'ID':<5} {'Class':<20} {'Confidence':<12} {'Bounding Box'}")
    print("=" * 90)

    for index, obj in enumerate(detections, start=1):

        x1, y1, x2, y2 = obj["bbox"]

        print(
            f"{index:<5}"
            f"{obj['class_name']:<20}"
            f"{obj['confidence']:.2f}      "
            f"({x1}, {y1}) ({x2}, {y2})"
        )

    print("=" * 90)