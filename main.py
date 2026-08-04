# =========================================================
# Industrial Visual AI Edge Node
#
# Run:
# python main.py
# =========================================================

import os


# =========================================================
# PADDLEPADDLE CPU COMPATIBILITY SETTINGS
#
# IMPORTANT:
# These environment variables must be configured before
# PaddleOCR or PaddlePaddle is imported.
#
# This disables oneDNN/MKLDNN optimization to avoid:
#
# ConvertPirAttribute2RuntimeAttribute not support
# [pir::ArrayAttribute<pir::DoubleAttribute>]
# =========================================================

os.environ["FLAGS_use_mkldnn"] = "0"

os.environ["FLAGS_enable_onednn"] = "0"

os.environ["FLAGS_pir_apply_inplace_pass"] = "0"


# =========================================================
# STANDARD IMPORTS
# =========================================================

import uvicorn

from datetime import datetime

from dotenv import load_dotenv


# =========================================================
# PROJECT IMPORTS
# =========================================================

from config import Config

from models.detector import (
    ObjectDetector
)

from communication.socket_client import (
    SocketClient
)

from communication.inspection_sender import (
    InspectionSender
)

from streaming.camera_processor import (
    CameraProcessor
)

from communication.webrtc_server import (
    WebRTCServer
)

from OCR.PaddleOCR.paddle_ocr_service import (
    PaddleOCRService
)


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# ENVIRONMENT CONFIGURATION
# =========================================================

SOCKET_SERVER_URL = os.getenv(
    "SOCKET_SERVER_URL"
)


# =========================================================
# CAMERA AI SETTINGS
# =========================================================

DEFAULT_AI_MODE = "object_detection"

OCR_INTERVAL_SECONDS = 2

DUPLICATE_INTERVAL_SECONDS = 10


# =========================================================
# MAIN APPLICATION
# =========================================================

def main():

    print(
        "\n"
        "========================================"
    )

    print(
        " Starting Industrial Visual AI Edge Node"
    )

    print(
        "========================================"
        "\n"
    )


    # =====================================================
    # DISPLAY PADDLE COMPATIBILITY SETTINGS
    # =====================================================

    print(
        "Paddle CPU compatibility mode: ENABLED"
    )

    print(
        "oneDNN/MKLDNN: DISABLED"
    )


    # =====================================================
    # LOAD CAMERA CONFIGURATION
    # =====================================================

    print(
        "\n"
        "Loading camera configuration..."
    )

    config = Config()

    cameras = config.cameras

    print(
        f"Configured cameras: "
        f"{len(cameras)}"
    )


    # =====================================================
    # VALIDATE SOCKET SERVER URL
    # =====================================================

    if not SOCKET_SERVER_URL:

        raise ValueError(

            "SOCKET_SERVER_URL is missing "
            "from the .env file"
        )


    print(
        "Socket.IO Backend URL:",
        SOCKET_SERVER_URL
    )


    # =====================================================
    # CREATE ONE SHARED SOCKET.IO CLIENT
    # =====================================================

    print(
        "\n"
        "Starting Socket.IO client..."
    )

    socket_client = SocketClient(

        SOCKET_SERVER_URL
    )

    socket_client.connect()


    # =====================================================
    # CREATE ONE SHARED INSPECTION SENDER
    # =====================================================

    print(
        "\n"
        "Creating shared inspection sender..."
    )

    inspection_sender = InspectionSender(

        socket_client=
            socket_client,

        site_id=
            "SITE-001",

        default_section_id=
            "SECTION-A"
    )

    print(
        "Shared inspection sender ready"
    )


    # =====================================================
    # SEND EDGE NODE STARTUP LOG
    # =====================================================

    startup_log = {

        "source":
            "edge_node",

        "level":
            "INFO",

        "message":
            (
                "Industrial Visual AI "
                "Edge Node started"
            ),

        "timestamp":
            datetime.now()
            .isoformat()
    }


    socket_client.send_log(

        startup_log
    )


    # =====================================================
    # CHECK REQUIRED AI SERVICES
    #
    # Camera AI services are loaded according to the
    # ai_mode configured in config/camera.json.
    # =====================================================

    enabled_cameras = [

        camera

        for camera in cameras

        if camera.get(

            "enabled",

            True
        )
    ]


    configured_ai_modes = [

        str(

            camera.get(

                "ai_mode",

                DEFAULT_AI_MODE
            )
        )

        .strip()

        .lower()

        for camera

        in enabled_cameras
    ]


    object_detector_required = any(

        ai_mode in [

            "object_detection",

            "both"
        ]

        for ai_mode

        in configured_ai_modes
    )


    ocr_service_required = any(

        ai_mode in [

            "ocr",

            "both"
        ]

        for ai_mode

        in configured_ai_modes
    )


    # =====================================================
    # LOAD ONE SHARED YOLO MODEL
    #
    # YOLO loads only when an enabled camera uses:
    #
    # object_detection
    # both
    # =====================================================

    detector = None


    if object_detector_required:

        print(
            "\n"
            "Loading shared object detector..."
        )

        detector = ObjectDetector()

        print(
            "Shared object detector ready"
        )

    else:

        print(
            "\n"
            "Shared object detector: "
            "NOT REQUIRED"
        )


    # =====================================================
    # LOAD ONE SHARED PADDLEOCR SERVICE
    #
    # PaddleOCR loads only when an enabled camera uses:
    #
    # ocr
    # both
    # =====================================================

    ocr_service = None


    if ocr_service_required:

        print(
            "\n"
            "Loading shared PaddleOCR service..."
        )


        ocr_service = PaddleOCRService(

            language=
                "en",

            confidence_threshold=
                0.60
        )


        print(
            "Shared PaddleOCR service ready"
        )

    else:

        print(
            "\n"
            "Shared PaddleOCR service: "
            "NOT REQUIRED"
        )


    # =====================================================
    # CREATE CAMERA PROCESSOR OBJECTS
    # =====================================================

    camera_processors = {}


    print(
        "\n"
        "Starting camera processors..."
        "\n"
    )


    for camera in cameras:


        # =================================================
        # SKIP DISABLED CAMERA
        # =================================================

        if not camera.get(

            "enabled",

            True

        ):

            print(

                "Skipping disabled camera:",

                camera["name"]
            )

            continue


        # =================================================
        # READ CAMERA CONFIGURATION
        # =================================================

        camera_id = (

            camera["id"]
        )


        camera_name = (

            camera["name"]
        )


        camera_source = (

            camera["source"]
        )


        camera_ai_mode = (

            str(

                camera.get(

                    "ai_mode",

                    DEFAULT_AI_MODE
                )
            )

            .strip()

            .lower()
        )


        # =================================================
        # DISPLAY CAMERA INFORMATION
        # =================================================

        print(

            f"Starting {camera_name}"
        )


        print(

            "Camera ID:",

            camera_id
        )


        print(

            "Camera source:",

            camera_source
        )


        print(

            "AI mode:",

            camera_ai_mode
        )


        # =================================================
        # DISPLAY CAMERA AI SERVICES
        # =================================================

        if camera_ai_mode == "ocr":

            print(
                "OCR: ENABLED"
            )

            print(
                "Object detection: DISABLED"
            )


        elif camera_ai_mode == "object_detection":

            print(
                "OCR: DISABLED"
            )

            print(
                "Object detection: ENABLED"
            )


        elif camera_ai_mode == "both":

            print(
                "OCR: ENABLED"
            )

            print(
                "Object detection: ENABLED"
            )


        elif camera_ai_mode == "none":

            print(
                "OCR: DISABLED"
            )

            print(
                "Object detection: DISABLED"
            )


        # =================================================
        # CREATE REUSABLE CAMERA PROCESSOR
        #
        # Each camera receives:
        #
        # camera_id
        # camera source
        # configured AI mode
        #
        # YOLO and PaddleOCR services are shared.
        # =================================================

        camera_processor = (

            CameraProcessor(

                camera_id=
                    camera_id,

                source=
                    camera_source,

                ai_mode=
                    camera_ai_mode,

                detector=
                    detector,

                ocr_service=
                    ocr_service,

                inspection_sender=
                    inspection_sender,

                ocr_interval_seconds=
                    OCR_INTERVAL_SECONDS,

                duplicate_interval_seconds=
                    DUPLICATE_INTERVAL_SECONDS
            )
        )


        # =================================================
        # START CAMERA PROCESSOR
        # =================================================

        camera_processor.start()


        # =================================================
        # SAVE CAMERA PROCESSOR
        # =================================================

        camera_processors[

            camera_id

        ] = (

            camera_processor
        )


        print(

            f"{camera_name} started"
        )


        print(

            "----------------------------------------"
        )


    # =====================================================
    # DISPLAY ACTIVE CAMERAS
    # =====================================================

    active_camera_ids = list(

        camera_processors.keys()
    )


    print(

        "\n"
        f"Active camera processors: "
        f"{len(camera_processors)}"
    )


    print(

        "Camera IDs:",

        active_camera_ids
    )


    # =====================================================
    # VALIDATE ACTIVE CAMERAS
    # =====================================================

    if not active_camera_ids:

        raise RuntimeError(

            "No active cameras are available."
        )


    # =====================================================
    # TEST GENERATOR DISABLED
    # =====================================================

    print(

        "\n"
        "Inspection test generator: DISABLED"
    )


    # =====================================================
    # CREATE MULTI-CAMERA WEBRTC SERVER
    # =====================================================

    print(

        "\n"
        "Creating multi-camera "
        "WebRTC server..."
    )


    webrtc_server = (

        WebRTCServer(

            camera_processors
        )
    )


    # =====================================================
    # DISPLAY WEBRTC SERVER INFORMATION
    # =====================================================

    print(

        "\n"
        "========================================"
    )


    print(
        "Edge Node WebRTC Server"
    )


    print(
        "Local URL:"
    )


    print(
        "http://localhost:8000"
    )


    print(
        "LAN URL:"
    )


    print(
        "http://192.168.0.180:8000"
    )


    print(
        "Health API:"
    )


    print(
        "http://192.168.0.180:8000/health"
    )


    print(
        "Camera API:"
    )


    print(
        "http://192.168.0.180:8000/cameras"
    )


    print(

        "========================================"
        "\n"
    )


    # =====================================================
    # RUN FASTAPI / UVICORN WEBRTC SERVER
    # =====================================================

    try:

        uvicorn.run(

            webrtc_server.app,

            host=
                "0.0.0.0",

            port=
                8000
        )


    # =====================================================
    # USER STOPPED APPLICATION
    # =====================================================

    except KeyboardInterrupt:

        print(

            "\n"
            "Edge Node stopped by user"
        )


    # =====================================================
    # EDGE NODE SHUTDOWN
    # =====================================================

    finally:

        print(

            "\n"
            "Stopping camera processors..."
        )


        for (

            camera_id,

            camera_processor

        ) in camera_processors.items():

            try:

                camera_processor.stop()

            except Exception as error:

                print(

                    "Camera shutdown error:",

                    camera_id,

                    str(error)
                )


        print(

            "Disconnecting "
            "Socket.IO client..."
        )


        socket_client.disconnect()


        print(

            "Edge Node shutdown complete"
        )


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()


# =========================================================