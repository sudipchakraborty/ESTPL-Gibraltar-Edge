# =========================================================
# Inspection Test Generator
#
# File:
# testing/inspection_test_generator.py
#
# Responsibility:
# Generate test inspection records from multiple cameras
# at a configurable interval.
# =========================================================

import threading
import time


# =========================================================
# INSPECTION TEST GENERATOR
# =========================================================

class InspectionTestGenerator:

    def __init__(
        self,
        inspection_sender,
        camera_ids,
        interval_seconds=5
    ):

        self.inspection_sender = (
            inspection_sender
        )

        self.camera_ids = (
            camera_ids
        )

        self.interval_seconds = (
            interval_seconds
        )

        self.running = False

        self.thread = None

        self.serial_number = 1


    # =====================================================
    # START TEST GENERATOR
    # =====================================================

    def start(self):

        if self.running:

            print(
                "[TEST GENERATOR] "
                "Already running"
            )

            return


        self.running = True


        self.thread = threading.Thread(

            target=
                self._run,

            daemon=
                True
        )


        self.thread.start()


        print(
            "[TEST GENERATOR] Started"
        )


        print(
            "[TEST GENERATOR] Cameras:",
            self.camera_ids
        )


        print(
            "[TEST GENERATOR] Interval:",
            self.interval_seconds,
            "seconds"
        )


    # =====================================================
    # GENERATOR LOOP
    # =====================================================

    def _run(self):

        while self.running:

            try:

                # Select camera in sequence
                camera_index = (

                    (
                        self.serial_number
                        - 1
                    )

                    %

                    len(
                        self.camera_ids
                    )
                )


                camera_id = (

                    self.camera_ids[
                        camera_index
                    ]
                )


                # Every third inspection fails
                is_pass = (

                    self.serial_number
                    % 3

                    !=

                    0
                )


                if is_pass:

                    self.inspection_sender.send_pass(

                        camera_id=
                            camera_id,

                        captured_data=(

                            f"GAS-2026-"

                            f"{self.serial_number:04d}"
                        ),

                        confidence=
                            0.96,

                        event_type=
                            "emboss_inspection",

                        comments=
                            "Emboss text matched",

                        remarks=
                            "Inspection completed"
                    )


                else:

                    self.inspection_sender.send_fail(

                        camera_id=
                            camera_id,

                        captured_data=
                            "TEXT-MISMATCH",

                        confidence=
                            0.72,

                        event_type=
                            "emboss_inspection",

                        comments=
                            (
                                "Emboss text "
                                "mismatch detected"
                            ),

                        remarks=
                            (
                                "Manual verification "
                                "required"
                            )
                    )


                self.serial_number += 1


            except Exception as error:

                print(

                    "[TEST GENERATOR ERROR]",

                    str(error)
                )


            time.sleep(

                self.interval_seconds
            )


    # =====================================================
    # STOP TEST GENERATOR
    # =====================================================

    def stop(self):

        self.running = False


        print(

            "[TEST GENERATOR] Stopped"
        )