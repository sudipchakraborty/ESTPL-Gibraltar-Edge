# =========================================================
# Reusable Inspection Sender
#
# File:
# communication/inspection_sender.py
#
# Responsibility:
# Create and send inspection events from any camera
# to the Node.js backend through the shared Socket.IO client.
# =========================================================

from datetime import (
    datetime,
    timezone
)

from threading import (
    Lock
)

import uuid


# =========================================================
# INSPECTION SENDER
# =========================================================

class InspectionSender:

    def __init__(
        self,
        socket_client,
        site_id="SITE-001",
        default_section_id=None
    ):

        # Shared Socket.IO client
        self.socket_client = (
            socket_client
        )

        # Edge Node site information
        self.site_id = (
            site_id
        )

        self.default_section_id = (
            default_section_id
        )

        # Protect event generation when multiple
        # camera threads send events simultaneously
        self.lock = Lock()

        print(
            "[INSPECTION SENDER] Ready"
        )

        print(
            "[INSPECTION SENDER] Site:",
            self.site_id
        )


    # =====================================================
    # GENERATE UNIQUE EVENT ID
    # =====================================================

    def generate_event_id(
        self,
        camera_id
    ):

        with self.lock:

            timestamp = (
                datetime.now(
                    timezone.utc
                )
                .strftime(
                    "%Y%m%d%H%M%S%f"
                )
            )

            unique_code = (
                uuid.uuid4()
                .hex[:8]
                .upper()
            )

            event_id = (

                f"{camera_id}-"

                f"{timestamp}-"

                f"{unique_code}"
            )

            return event_id


    # =====================================================
    # SEND INSPECTION
    # =====================================================

    def send(
        self,

        camera_id,

        status,

        captured_data=None,

        section_id=None,

        event_type=(
            "visual_inspection"
        ),

        evidence_link=None,

        comments=None,

        remarks=None,

        confidence=None,

        timestamp=None,

        event_id=None
    ):

        try:

            # =============================================
            # VALIDATE CAMERA ID
            # =============================================

            if not camera_id:

                raise ValueError(
                    "camera_id is required"
                )


            # =============================================
            # NORMALIZE STATUS
            # =============================================

            normalized_status = (

                str(status)
                .strip()
                .upper()
            )


            valid_status_values = [

                "PASS",

                "FAIL",

                "WARNING",

                "UNKNOWN"
            ]


            if (
                normalized_status
                not in
                valid_status_values
            ):

                raise ValueError(

                    "Invalid status: "

                    f"{normalized_status}"
                )


            # =============================================
            # GENERATE EVENT ID
            # =============================================

            if not event_id:

                event_id = (

                    self
                    .generate_event_id(
                        camera_id
                    )
                )


            # =============================================
            # CREATE DATABASE-COMPATIBLE PAYLOAD
            # =============================================

            inspection_data = {

                "site_id":
                    self.site_id,

                "section_id":
                    (
                        section_id

                        or

                        self
                        .default_section_id
                    ),

                "camera_id":
                    camera_id,

                "captured_data":
                    captured_data,

                "event_id":
                    event_id,

                "event_type":
                    event_type,

                "status":
                    normalized_status,

                "evidence_link":
                    evidence_link,

                "comments":
                    comments,

                "remarks":
                    remarks,

                "confidence":
                    confidence,

                "timestamp":
                    (
                        timestamp

                        or

                        datetime.now(
                            timezone.utc
                        )
                        .isoformat()
                    )
            }


            # =============================================
            # SEND THROUGH SHARED SOCKET.IO CLIENT
            # =============================================

            sent = (

                self
                .socket_client
                .send_inspection_status(

                    inspection_data
                )
            )


            # =============================================
            # LOG RESULT
            # =============================================

            if sent:

                print(

                    "[INSPECTION SENT]",

                    f"Camera={camera_id}",

                    f"Event={event_id}",

                    (
                        "Status="
                        f"{normalized_status}"
                    )
                )

            else:

                print(

                    "[INSPECTION NOT SENT]",

                    f"Camera={camera_id}",

                    f"Event={event_id}"
                )


            # Return useful information
            return {

                "success":
                    bool(sent),

                "event_id":
                    event_id,

                "data":
                    inspection_data
            }


        except Exception as error:

            print(

                "[INSPECTION SENDER ERROR]",

                str(error)
            )


            return {

                "success":
                    False,

                "event_id":
                    event_id,

                "error":
                    str(error)
            }


    # =====================================================
    # SEND PASS INSPECTION
    # =====================================================

    def send_pass(

        self,

        camera_id,

        captured_data=None,

        confidence=None,

        comments=(
            "Visual inspection passed"
        ),

        remarks=(
            "Inspection completed"
        ),

        **additional_data
    ):

        return self.send(

            camera_id=
                camera_id,

            status=
                "PASS",

            captured_data=
                captured_data,

            confidence=
                confidence,

            comments=
                comments,

            remarks=
                remarks,

            **additional_data
        )


    # =====================================================
    # SEND FAIL INSPECTION
    # =====================================================

    def send_fail(

        self,

        camera_id,

        captured_data=None,

        confidence=None,

        comments=(
            "Visual inspection failed"
        ),

        remarks=(
            "Manual verification required"
        ),

        **additional_data
    ):

        return self.send(

            camera_id=
                camera_id,

            status=
                "FAIL",

            captured_data=
                captured_data,

            confidence=
                confidence,

            comments=
                comments,

            remarks=
                remarks,

            **additional_data
        )