import cv2
from loguru import logger

class VideoProcessor:
    def __init__(self, source, max_count=180):
        self.video_path = source.video_path
        self.rtsp = source.rtsp
        self.frame_count = 0
        self.max_count = max_count
        self.index = 0
        self.fps = None

    def __enter__(self):
        if self.video_path and self.rtsp.label is False:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap:
                logger.error("can not open the video")
            self.frame_count = self.cap.get(int(cv2.CAP_PROP_FRAME_COUNT))
            if self.max_count and self.max_count > 0:
                self.frame_count = min(self.frame_count, self.max_count)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        elif self.rtsp.label:
            account = self.rtsp.account
            password = self.rtsp.password
            ip_address = self.rtsp.ip_address
            channel = int(self.rtsp.channel)
            camera = "rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" \
                                % (account, password, ip_address, channel)
            self.cap = cv2.VideoCapture(camera)
        else:
            logger.error("source video stream error")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == self.frame_count:
            return None

        _ret, _frame = self.cap.read()
        if not _ret:
            return None

        self.index += 1
        return _frame

    def __len__(self):
        return self.frame_count