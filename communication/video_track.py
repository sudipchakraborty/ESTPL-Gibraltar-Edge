import asyncio
import av

from aiortc import VideoStreamTrack


class ProcessedVideoTrack(VideoStreamTrack):

    def __init__(self, camera_processor):

        super().__init__()

        self.camera_processor = camera_processor


    async def recv(self):

        # WebRTC timestamp
        pts, time_base = await self.next_timestamp()

        frame = self.camera_processor.get_latest_frame()


        # Wait until first processed frame is available
        while frame is None:

            await asyncio.sleep(0.01)

            frame = (
                self.camera_processor.get_latest_frame()
            )


        # OpenCV BGR frame -> AV VideoFrame
        video_frame = av.VideoFrame.from_ndarray(
            frame,
            format="bgr24"
        )

        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame