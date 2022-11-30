import cv2


class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.frame_count = 0
        self.index = 0

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_count = self.cap.get(int(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 100:
            return None

        _ret, _frame = self.cap.read()
        if not _ret:
            return None

        self.index += 1
        return _frame

    def __len__(self):
        return self.frame_count