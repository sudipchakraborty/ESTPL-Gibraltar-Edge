import asyncio
import logging
import time
from fractions import Fraction
import cv2
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from aiortc import (RTCPeerConnection, RTCSessionDescription,VideoStreamTrack)
from av import VideoFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepVision-WebRTC")
# =========================================================
# CAMERA VIDEO TRACK
# =========================================================

class CameraVideoTrack(VideoStreamTrack):

    def __init__(self,camera_processor,camera_id,fps=10,target_width=960):
        super().__init__()
        self.camera_processor = camera_processor
        self.camera_id = camera_id
        self.fps = max(1.0, float(fps))
        self.target_width = max(320, int(target_width))
        self._pts = 0
        self._next_frame_at = time.monotonic()
        
    async def recv(self):

        # -------------------------------------------------
        # Generate WebRTC timestamp
        # -------------------------------------------------

        delay = self._next_frame_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        self._next_frame_at = max(self._next_frame_at + 1 / self.fps, time.monotonic())
        pts = self._pts
        time_base = Fraction(1, 90_000)
        self._pts += round(90_000 / self.fps)


        # -------------------------------------------------
        # Get the latest processed camera frame
        # -------------------------------------------------

        frame = (
            self.camera_processor
            .get_latest_frame()
        )


        # -------------------------------------------------
        # Wait until the camera produces a valid frame
        # -------------------------------------------------

        while frame is None:

            await asyncio.sleep(0.01)

            frame = (
                self.camera_processor
                .get_latest_frame()
            )


        # -------------------------------------------------
        # Convert OpenCV BGR frame to WebRTC VideoFrame
        # -------------------------------------------------

        height, width = frame.shape[:2]
        if width > self.target_width:
            target_height = max(2, round(height * self.target_width / width))
            frame = cv2.resize(frame, (self.target_width, target_height), interpolation=cv2.INTER_AREA)

        video_frame = VideoFrame.from_ndarray(

            frame,

            format="bgr24"

        )


        # -------------------------------------------------
        # Add WebRTC timing information
        # -------------------------------------------------

        video_frame.pts = pts

        video_frame.time_base = time_base


        return video_frame


# =========================================================
# WEBRTC SERVER
# =========================================================

class WebRTCServer:

    def __init__(
        self,
        camera_processors,
        fps=10,
        target_width=960,
    ):

        """
        camera_processors format:

        {
            "camera_1": CameraProcessor(...),
            "camera_2": CameraProcessor(...),
            "camera_3": CameraProcessor(...)
        }
        """


        # -------------------------------------------------
        # Store all camera processors
        # -------------------------------------------------

        self.camera_processors = (
            camera_processors
        )
        self.fps = fps
        self.target_width = target_width


        # -------------------------------------------------
        # Store active WebRTC connections
        # -------------------------------------------------

        self.peer_connections = set()


        # -------------------------------------------------
        # Create FastAPI application
        # -------------------------------------------------

        self.app = FastAPI(

            title=(
                "DeepVision "
                "Multi-Camera "
                "WebRTC Server"
            ),

            version="1.0.0"

        )


        # =================================================
        # CORS CONFIGURATION
        # =================================================

        self.app.add_middleware(

            CORSMiddleware,

            allow_origins=[
                "*"
            ],

            allow_credentials=False,

            allow_methods=[
                "*"
            ],

            allow_headers=[
                "*"
            ]

        )


        # =================================================
        # REGISTER API ROUTES
        # =================================================

        self.app.post(
            "/offer"
        )(
            self.offer
        )


        self.app.get(
            "/health"
        )(
            self.health
        )


        self.app.get(
            "/cameras"
        )(
            self.get_cameras
        )


        # =================================================
        # IMPORTANT
        # =================================================
        #
        # Do not use:
        #
        # self.app.add_event_handler(...)
        #
        # because it is unavailable in your installed
        # FastAPI version.
        #


    # =====================================================
    # WEBRTC OFFER ENDPOINT
    # =====================================================

    async def offer(
        self,
        request: Request
    ):

        peer_connection = None


        try:

            # -------------------------------------------------
            # Read request from React
            # -------------------------------------------------

            params = await request.json()


            # -------------------------------------------------
            # Read camera ID
            # -------------------------------------------------

            camera_id = params.get(

                "camera_id"

            )


            # -------------------------------------------------
            # Validate camera ID
            # -------------------------------------------------

            if camera_id is None:

                return {

                    "error": (
                        "camera_id "
                        "is required"
                    ),

                    "available_cameras": (

                        list(

                            self
                            .camera_processors
                            .keys()

                        )

                    )

                }


            # -------------------------------------------------
            # Check whether requested camera exists
            # -------------------------------------------------

            if (

                camera_id

                not in

                self.camera_processors

            ):

                return {

                    "error": (

                        f"Camera "
                        f"'{camera_id}' "
                        f"was not found"

                    ),

                    "available_cameras": (

                        list(

                            self
                            .camera_processors
                            .keys()

                        )

                    )

                }


            # -------------------------------------------------
            # Validate SDP
            # -------------------------------------------------

            if "sdp" not in params:

                return {

                    "error": (
                        "WebRTC SDP "
                        "is required"
                    )

                }


            # -------------------------------------------------
            # Validate WebRTC description type
            # -------------------------------------------------

            if "type" not in params:

                return {

                    "error": (
                        "WebRTC description "
                        "type is required"
                    )

                }


            # -------------------------------------------------
            # Get selected camera processor
            # -------------------------------------------------

            camera_processor = (

                self.camera_processors[

                    camera_id

                ]

            )


            logger.info(

                "WebRTC request received "
                "for camera: %s",

                camera_id

            )


            # -------------------------------------------------
            # Create new WebRTC peer connection
            # -------------------------------------------------

            peer_connection = (

                RTCPeerConnection()

            )


            # -------------------------------------------------
            # Store peer connection
            # -------------------------------------------------

            self.peer_connections.add(

                peer_connection

            )


            # -------------------------------------------------
            # Monitor connection state
            # -------------------------------------------------

            @peer_connection.on(

                "connectionstatechange"

            )

            async def connection_state_change():

                state = (

                    peer_connection
                    .connectionState

                )


                logger.info(

                    "Camera %s "
                    "WebRTC state: %s",

                    camera_id,

                    state

                )


                # Close failed or closed connections

                if state in [

                    "failed",

                    "closed"

                ]:

                    await (

                        peer_connection
                        .close()

                    )


                    self.peer_connections.discard(

                        peer_connection

                    )


            # -------------------------------------------------
            # Create browser offer
            # -------------------------------------------------

            browser_offer = (

                RTCSessionDescription(

                    sdp=params["sdp"],

                    type=params["type"]

                )

            )


            # -------------------------------------------------
            # Set browser SDP offer
            # -------------------------------------------------

            await (

                peer_connection
                .setRemoteDescription(

                    browser_offer

                )

            )


            # -------------------------------------------------
            # Create video track for selected camera
            # -------------------------------------------------

            camera_video_track = (

                CameraVideoTrack(

                    camera_processor=(

                        camera_processor

                    ),

                    camera_id=(

                        camera_id

                    ),

                    fps=self.fps,

                    target_width=self.target_width,

                )

            )


            # -------------------------------------------------
            # Add camera video to WebRTC
            # -------------------------------------------------

            peer_connection.addTrack(

                camera_video_track

            )


            # -------------------------------------------------
            # Create WebRTC answer
            # -------------------------------------------------

            answer = await (

                peer_connection
                .createAnswer()

            )


            # -------------------------------------------------
            # Set local WebRTC answer
            # -------------------------------------------------

            await (

                peer_connection
                .setLocalDescription(

                    answer

                )

            )


            logger.info(

                "WebRTC answer created "
                "for camera: %s",

                camera_id

            )


            # -------------------------------------------------
            # Return WebRTC answer to React
            # -------------------------------------------------

            return {

                "sdp": (

                    peer_connection
                    .localDescription
                    .sdp

                ),

                "type": (

                    peer_connection
                    .localDescription
                    .type

                )

            }


        except Exception as error:

            logger.exception(

                "WebRTC connection error"

            )


            # -------------------------------------------------
            # Close connection if initialization failed
            # -------------------------------------------------

            if peer_connection is not None:

                try:

                    await (

                        peer_connection
                        .close()

                    )

                except Exception:

                    pass


                self.peer_connections.discard(

                    peer_connection

                )


            return {

                "error": str(

                    error

                )

            }


    # =====================================================
    # GET CAMERA LIST
    # =====================================================

    async def get_cameras(
        self
    ):

        return {

            "camera_count": (

                len(

                    self
                    .camera_processors

                )

            ),

            "cameras": (

                list(

                    self
                    .camera_processors
                    .keys()

                )

            )

        }


    # =====================================================
    # HEALTH CHECK
    # =====================================================

    async def health(
        self
    ):

        return {

            "status": "running",

            "service": (

                "DeepVision "
                "Multi-Camera "
                "WebRTC Server"

            ),

            "camera_count": (

                len(

                    self
                    .camera_processors

                )

            ),

            "cameras": (

                list(

                    self
                    .camera_processors
                    .keys()

                )

            ),

            "active_webrtc_connections": (

                len(

                    self
                    .peer_connections

                )

            )

        }


    # =====================================================
    # MANUAL SHUTDOWN METHOD
    # =====================================================

    async def shutdown(
        self
    ):

        logger.info(

            "Closing all active "
            "WebRTC connections"

        )


        # -------------------------------------------------
        # Create close tasks
        # -------------------------------------------------

        close_tasks = [

            peer_connection.close()

            for peer_connection

            in self.peer_connections

        ]


        # -------------------------------------------------
        # Close all connections
        # -------------------------------------------------

        if close_tasks:

            await asyncio.gather(

                *close_tasks,

                return_exceptions=True

            )


        # -------------------------------------------------
        # Clear connection list
        # -------------------------------------------------

        self.peer_connections.clear()


        logger.info(

            "All WebRTC "
            "connections closed"

        )
