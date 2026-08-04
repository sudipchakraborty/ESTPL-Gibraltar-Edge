# =========================================================
# Reusable Non-Blocking Camera Processor
#
# File:
# streaming/camera_processor.py
#
# Supported AI modes:
#
# object_detection
# ocr
# both
# none
#
# Architecture:
#
# Camera Thread
#     -> Continuously captures frames
#     -> Updates WebRTC video
#     -> Draws cached AI results
#
# OCR Worker Thread
#     -> Runs PaddleOCR independently
#     -> Does not block camera capture
#     -> Processes only the newest available frame
#
# =========================================================

import cv2
import threading
import time
import numpy as np


# =========================================================
# CAMERA PROCESSOR
# =========================================================

class CameraProcessor:

    def __init__(
        self,
        camera_id,
        source,
        ai_mode="object_detection",
        detector=None,
        ocr_service=None,
        inspection_sender=None,
        ocr_interval_seconds=2,
        duplicate_interval_seconds=10
    ):

        # =================================================
        # CAMERA CONFIGURATION
        # =================================================

        self.camera_id = camera_id

        self.source = source

        self.ai_mode = (
            str(ai_mode)
            .strip()
            .lower()
        )


        # =================================================
        # SHARED AI SERVICES
        # =================================================

        self.detector = detector

        self.ocr_service = ocr_service


        # =================================================
        # BACKEND INSPECTION SENDER
        # =================================================

        self.inspection_sender = (
            inspection_sender
        )


        # =================================================
        # CAMERA OBJECT
        # =================================================

        self.cap = None


        # =================================================
        # LATEST CAMERA DATA
        # =================================================

        self.latest_raw_frame = None

        self.latest_frame = None

        self.latest_detections = []

        self.latest_ocr_results = []


        # =================================================
        # THREAD CONFIGURATION
        # =================================================

        self.running = False

        self.camera_thread = None

        self.ocr_thread = None


        # =================================================
        # THREAD LOCKS
        # =================================================

        # Protects frames and AI results.

        self.lock = threading.Lock()


        # =================================================
        # OCR WORKER EVENT
        #
        # The camera thread signals this event when OCR
        # should process a new frame.
        # =================================================

        self.ocr_event = threading.Event()


        # =================================================
        # OCR WORKER STATE
        # =================================================

        self.ocr_processing = False


        # =================================================
        # OCR TIMING
        # =================================================

        self.ocr_interval_seconds = float(

            ocr_interval_seconds
        )


        self.last_ocr_request_time = 0.0


        # =================================================
        # DUPLICATE DATABASE PROTECTION
        # =================================================

        self.duplicate_interval_seconds = float(

            duplicate_interval_seconds
        )


        self.last_sent_text = ""

        self.last_sent_time = 0.0


        # =================================================
        # VALIDATE AI MODE
        # =================================================

        valid_ai_modes = [

            "object_detection",

            "ocr",

            "both",

            "none"
        ]


        if self.ai_mode not in valid_ai_modes:

            raise ValueError(

                f"Invalid AI mode "
                f"'{self.ai_mode}' "
                f"for camera "
                f"'{self.camera_id}'. "
                f"Supported modes: "
                f"{valid_ai_modes}"
            )


        # =================================================
        # VALIDATE OBJECT DETECTOR
        # =================================================

        if (

            self.ai_mode

            in [

                "object_detection",

                "both"
            ]

            and

            self.detector is None

        ):

            raise ValueError(

                "Object detector is required "
                f"for camera "
                f"{self.camera_id}"
            )


        # =================================================
        # VALIDATE OCR SERVICE
        # =================================================

        if (

            self.ai_mode

            in [

                "ocr",

                "both"
            ]

            and

            self.ocr_service is None

        ):

            raise ValueError(

                "OCR service is required "
                f"for camera "
                f"{self.camera_id}"
            )


        # =================================================
        # STARTUP INFORMATION
        # =================================================

        print(

            "[CAMERA PROCESSOR] Created"
        )


        print(

            "[CAMERA PROCESSOR] Camera ID:",

            self.camera_id
        )


        print(

            "[CAMERA PROCESSOR] AI mode:",

            self.ai_mode
        )


        print(

            "[CAMERA PROCESSOR] OCR interval:",

            self.ocr_interval_seconds,

            "seconds"
        )


    # =====================================================
    # START CAMERA PROCESSOR
    # =====================================================

    def start(self):

        if self.running:

            return


        print(

            f"Opening camera stream: "
            f"{self.source}"
        )


        # =================================================
        # OPEN CAMERA
        # =================================================

        self.cap = cv2.VideoCapture(

            self.source
        )


        if not self.cap.isOpened():

            raise RuntimeError(

                "Cannot open camera source: "

                f"{self.source}"
            )


        # =================================================
        # REDUCE CAMERA BUFFERING
        #
        # Some OpenCV camera backends may ignore this
        # setting. It is safe if unsupported.
        # =================================================

        try:

            self.cap.set(

                cv2.CAP_PROP_BUFFERSIZE,

                1
            )

        except Exception:

            pass


        # =================================================
        # ENABLE PROCESSING
        # =================================================

        self.running = True


        # =================================================
        # CREATE CAMERA CAPTURE THREAD
        # =================================================

        self.camera_thread = threading.Thread(

            target=
                self._camera_loop,

            name=
                (
                    f"CameraThread-"
                    f"{self.camera_id}"
                ),

            daemon=
                True
        )


        self.camera_thread.start()


        # =================================================
        # CREATE OCR WORKER THREAD
        # =================================================

        if (

            self.ai_mode

            in [

                "ocr",

                "both"
            ]

        ):

            self.ocr_thread = threading.Thread(

                target=
                    self._ocr_worker_loop,

                name=
                    (
                        f"OCRThread-"
                        f"{self.camera_id}"
                    ),

                daemon=
                    True
            )


            self.ocr_thread.start()


            print(

                "[OCR WORKER] Started",

                f"Camera={self.camera_id}"
            )


        print(

            "Camera processor started"
        )


    # =====================================================
    # MAIN CAMERA LOOP
    #
    # This thread never waits for PaddleOCR.
    # =====================================================

    def _camera_loop(self):

        while self.running:


            # =============================================
            # READ CAMERA FRAME
            # =============================================

            ret, frame = (

                self.cap.read()
            )


            if not ret:

                print(

                    "[CAMERA READ ERROR]",

                    self.camera_id
                )


                time.sleep(

                    0.05
                )


                continue


            try:

                # =========================================
                # SAVE THE NEWEST RAW CAMERA FRAME
                #
                # OCR worker always uses the newest frame.
                # =========================================

                with self.lock:

                    self.latest_raw_frame = (

                        frame.copy()
                    )


                # =========================================
                # CREATE DISPLAY FRAME
                # =========================================

                processed_frame = (

                    frame.copy()
                )


                # =========================================
                # OBJECT DETECTION
                #
                # NOTE:
                # YOLO is still synchronous in this
                # version. OCR is fully non-blocking.
                # =========================================

                if (

                    self.ai_mode

                    in [

                        "object_detection",

                        "both"
                    ]

                ):

                    processed_frame = (

                        self
                        ._process_object_detection(

                            processed_frame
                        )
                    )


                # =========================================
                # REQUEST OCR
                #
                # This only signals the OCR worker.
                # It does not run OCR in this thread.
                # =========================================

                if (

                    self.ai_mode

                    in [

                        "ocr",

                        "both"
                    ]

                ):

                    self._request_ocr_if_required()


                    # Draw cached OCR results while the
                    # next OCR operation runs separately.

                    processed_frame = (

                        self
                        ._draw_ocr_results(

                            processed_frame
                        )
                    )


                # =========================================
                # NO AI MODE
                # =========================================

                if self.ai_mode == "none":

                    with self.lock:

                        self.latest_detections = []

                        self.latest_ocr_results = []


                # =========================================
                # UPDATE WEBRTC DISPLAY FRAME
                # =========================================

                with self.lock:

                    self.latest_frame = (

                        processed_frame.copy()
                    )


            except Exception as error:

                print(

                    "[FRAME PROCESSING ERROR]",

                    f"Camera={self.camera_id}",

                    str(error)
                )


    # =====================================================
    # REQUEST OCR WHEN INTERVAL EXPIRES
    # =====================================================

    def _request_ocr_if_required(self):

        current_time = time.time()


        # =================================================
        # OCR INTERVAL HAS NOT EXPIRED
        # =================================================

        if (

            current_time

            -

            self.last_ocr_request_time

            <

            self.ocr_interval_seconds

        ):

            return


        # =================================================
        # PREVIOUS OCR IS STILL RUNNING
        #
        # Do not queue multiple old camera frames.
        # =================================================

        if self.ocr_processing:

            return


        # =================================================
        # OCR REQUEST ALREADY EXISTS
        # =================================================

        if self.ocr_event.is_set():

            return


        # =================================================
        # SAVE REQUEST TIME
        # =================================================

        self.last_ocr_request_time = (

            current_time
        )


        # =================================================
        # WAKE OCR WORKER
        # =================================================

        self.ocr_event.set()


    # =====================================================
    # OCR WORKER LOOP
    #
    # PaddleOCR runs only in this worker thread.
    # =====================================================

    def _ocr_worker_loop(self):

        while self.running:


            # =============================================
            # WAIT FOR OCR REQUEST
            #
            # Timeout allows clean shutdown.
            # =============================================

            request_received = (

                self.ocr_event.wait(

                    timeout=
                        0.25
                )
            )


            if not self.running:

                break


            if not request_received:

                continue


            # =============================================
            # REMOVE CURRENT REQUEST
            # =============================================

            self.ocr_event.clear()


            # =============================================
            # COPY ONLY THE NEWEST CAMERA FRAME
            # =============================================

            with self.lock:

                if self.latest_raw_frame is None:

                    continue


                ocr_frame = (

                    self.latest_raw_frame.copy()
                )


            # =============================================
            # RUN OCR OUTSIDE THE CAMERA THREAD
            # =============================================

            self.ocr_processing = True


            try:

                self._process_ocr(

                    ocr_frame
                )


            except Exception as error:

                print(

                    "[OCR WORKER ERROR]",

                    f"Camera={self.camera_id}",

                    str(error)
                )


            finally:

                self.ocr_processing = False


    # =====================================================
    # OBJECT DETECTION
    # =====================================================

    def _process_object_detection(

        self,

        frame

    ):

        try:

            # =============================================
            # RUN OBJECT DETECTOR
            # =============================================

            detections, results = (

                self.detector.detect(

                    frame
                )
            )


            # =============================================
            # DRAW OBJECT ANNOTATIONS
            # =============================================

            processed_frame = (

                self.detector.draw(

                    frame,

                    results
                )
            )


            # =============================================
            # SAVE LATEST DETECTIONS
            # =============================================

            with self.lock:

                self.latest_detections = (

                    detections
                )


            return processed_frame


        except Exception as error:

            print(

                "[OBJECT DETECTION ERROR]",

                f"Camera={self.camera_id}",

                str(error)
            )


            return frame


    # =====================================================
    # OCR PROCESSING
    # =====================================================

    def _process_ocr(

        self,

        frame

    ):

        try:

            # =============================================
            # RECORD OCR START TIME
            # =============================================

            ocr_start_time = (

                time.time()
            )


            # =============================================
            # RUN PADDLEOCR
            # =============================================

            ocr_results = (

                self.ocr_service.read(

                    frame
                )
            )


            # =============================================
            # CALCULATE OCR PROCESSING TIME
            # =============================================

            ocr_duration = (

                time.time()

                -

                ocr_start_time
            )


            print(

                "[OCR PERFORMANCE]",

                f"Camera={self.camera_id}",

                (
                    f"Time="
                    f"{ocr_duration:.2f}s"
                )
            )


            # =============================================
            # SAVE LATEST OCR RESULTS
            # =============================================

            with self.lock:

                self.latest_ocr_results = (

                    list(

                        ocr_results
                    )
                )


            # =============================================
            # NO OCR TEXT DETECTED
            # =============================================

            if not ocr_results:

                print(

                    "[OCR] No text detected",

                    f"Camera={self.camera_id}"
                )


                return


            # =============================================
            # COMBINE ALL OCR TEXT
            # =============================================

            detected_text = (

                " ".join(

                    [

                        str(

                            result.get(

                                "text",

                                ""
                            )
                        )

                        .strip()

                        for result

                        in ocr_results

                        if result.get(

                            "text"
                        )

                    ]
                )
            )


            detected_text = (

                detected_text.strip()
            )


            # =============================================
            # CALCULATE AVERAGE OCR CONFIDENCE
            # =============================================

            scores = [

                float(

                    result.get(

                        "score",

                        0.0
                    )
                )

                for result

                in ocr_results
            ]


            average_confidence = (

                sum(scores)

                /

                len(scores)

                if scores

                else

                0.0
            )


            # =============================================
            # DISPLAY OCR RESULT
            # =============================================

            print(

                "[OCR TEXT]",

                f"Camera={self.camera_id}",

                f"Text={detected_text}",

                (
                    "Confidence="
                    f"{average_confidence:.2f}"
                )
            )


            # =============================================
            # SEND OCR RESULT TO BACKEND
            # =============================================

            self._send_ocr_inspection(

                detected_text=
                    detected_text,

                confidence=
                    average_confidence
            )


        except Exception as error:

            print(

                "[OCR PROCESSING ERROR]",

                f"Camera={self.camera_id}",

                str(error)
            )


    # =====================================================
    # DRAW CACHED OCR RESULTS
    # =====================================================

    def _draw_ocr_results(

        self,

        frame

    ):

        try:

            # =============================================
            # COPY OCR RESULTS
            # =============================================

            with self.lock:

                ocr_results = [

                    dict(result)

                    for result

                    in self.latest_ocr_results
                ]


            # =============================================
            # DRAW EACH OCR RESULT
            # =============================================

            for result in ocr_results:


                text = str(

                    result.get(

                        "text",

                        ""
                    )
                )


                score = float(

                    result.get(

                        "score",

                        0.0
                    )
                )


                box = (

                    result.get(

                        "box"
                    )
                )


                if box is None:

                    continue


                # =========================================
                # CONVERT OCR BOX TO NUMPY ARRAY
                # =========================================

                points = np.array(

                    box,

                    dtype=
                        np.int32
                )


                # =========================================
                # SUPPORT RECTANGULAR BOX FORMAT
                #
                # [x1, y1, x2, y2]
                # =========================================

                if (

                    points.ndim == 1

                    and

                    len(points) == 4

                ):

                    x1, y1, x2, y2 = (

                        points.tolist()
                    )


                    points = np.array(

                        [

                            [x1, y1],

                            [x2, y1],

                            [x2, y2],

                            [x1, y2]

                        ],

                        dtype=
                            np.int32
                    )


                # =========================================
                # VALIDATE POLYGON FORMAT
                # =========================================

                if (

                    points.ndim != 2

                    or

                    points.shape[1] != 2

                ):

                    continue


                # =========================================
                # DRAW OCR BOUNDING BOX
                # =========================================

                cv2.polylines(

                    frame,

                    [

                        points
                    ],

                    True,

                    (
                        0,

                        255,

                        0
                    ),

                    2
                )


                # =========================================
                # OCR LABEL POSITION
                # =========================================

                label_x = max(

                    5,

                    int(

                        points[0][0]
                    )
                )


                label_y = max(

                    20,

                    int(

                        points[0][1]
                    )

                    -

                    8
                )


                # =========================================
                # SHORT DISPLAY LABEL
                #
                # Removing "OCR:" reduces overlap.
                # =========================================

                label = (

                    f"{text} "

                    f"{score:.2f}"
                )


                # =========================================
                # DRAW TEXT BACKGROUND
                # =========================================

                (

                    text_width,

                    text_height

                ), baseline = (

                    cv2.getTextSize(

                        label,

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.50,

                        1
                    )
                )


                background_x2 = min(

                    frame.shape[1] - 1,

                    label_x

                    +

                    text_width

                    +

                    6
                )


                background_y1 = max(

                    0,

                    label_y

                    -

                    text_height

                    -

                    6
                )


                cv2.rectangle(

                    frame,

                    (

                        label_x,

                        background_y1
                    ),

                    (

                        background_x2,

                        label_y

                        +

                        baseline
                    ),

                    (

                        0,

                        0,

                        0
                    ),

                    -1
                )


                # =========================================
                # DRAW OCR TEXT
                # =========================================

                cv2.putText(

                    frame,

                    label,

                    (

                        label_x

                        +

                        3,

                        label_y

                        -

                        2
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.50,

                    (

                        0,

                        255,

                        0
                    ),

                    1,

                    cv2.LINE_AA
                )


            return frame


        except Exception as error:

            print(

                "[OCR DRAW ERROR]",

                f"Camera={self.camera_id}",

                str(error)
            )


            return frame


    # =====================================================
    # SEND OCR INSPECTION TO BACKEND
    # =====================================================

    def _send_ocr_inspection(

        self,

        detected_text,

        confidence

    ):

        # =============================================
        # VALIDATE INSPECTION SENDER
        # =============================================

        if self.inspection_sender is None:

            print(

                "[OCR] Inspection sender "
                "is not configured",

                f"Camera={self.camera_id}"
            )


            return


        # =============================================
        # VALIDATE OCR TEXT
        # =============================================

        if not detected_text:

            return


        current_time = (

            time.time()
        )


        # =============================================
        # DUPLICATE TEXT PROTECTION
        # =============================================

        is_same_text = (

            detected_text

            ==

            self.last_sent_text
        )


        duplicate_time_active = (

            current_time

            -

            self.last_sent_time

            <

            self.duplicate_interval_seconds
        )


        if (

            is_same_text

            and

            duplicate_time_active

        ):

            print(

                "[OCR] Duplicate text skipped",

                f"Camera={self.camera_id}",

                f"Text={detected_text}"
            )


            return


        # =============================================
        # SEND DATABASE-COMPATIBLE INSPECTION
        # =============================================

        result = (

            self
            .inspection_sender
            .send(

                camera_id=
                    self.camera_id,

                event_type=
                    "ocr_text_detection",

                status=
                    "PASS",

                captured_data=
                    detected_text,

                confidence=
                    round(

                        confidence,

                        4
                    ),

                comments=
                    "OCR text detected",

                remarks=
                    "OCR inspection completed"
            )
        )


        # =============================================
        # UPDATE DUPLICATE TRACKING
        # =============================================

        if result.get(

            "success",

            False

        ):

            self.last_sent_text = (

                detected_text
            )


            self.last_sent_time = (

                current_time
            )


    # =====================================================
    # GET LATEST PROCESSED FRAME
    # =====================================================

    def get_latest_frame(self):

        with self.lock:

            if self.latest_frame is None:

                return None


            return (

                self.latest_frame.copy()
            )


    # =====================================================
    # GET LATEST OBJECT DETECTIONS
    # =====================================================

    def get_latest_detections(self):

        with self.lock:

            return list(

                self.latest_detections
            )


    # =====================================================
    # GET LATEST OCR RESULTS
    # =====================================================

    def get_latest_ocr_results(self):

        with self.lock:

            return [

                dict(result)

                for result

                in self.latest_ocr_results
            ]


    # =====================================================
    # STOP CAMERA PROCESSOR
    # =====================================================

    def stop(self):

        print(

            "Stopping camera processor:",

            self.camera_id
        )


        # =================================================
        # STOP ALL THREADS
        # =================================================

        self.running = False


        # Wake the OCR worker so it can exit.

        self.ocr_event.set()


        # =================================================
        # WAIT FOR CAMERA THREAD
        # =================================================

        if self.camera_thread:

            self.camera_thread.join(

                timeout=
                    3
            )


        # =================================================
        # WAIT FOR OCR THREAD
        # =================================================

        if self.ocr_thread:

            self.ocr_thread.join(

                timeout=
                    5
            )


        # =================================================
        # RELEASE CAMERA
        # =================================================

        if self.cap:

            self.cap.release()


        print(

            "Camera processor stopped:",

            self.camera_id
        )