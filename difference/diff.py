import cv2
import imutils
from loguru import logger


class DiffProcessor:

    def get_frame_feature(self, frame):
        """Extract feature of frame."""
        raise NotImplementedError()

    def cal_frame_diff(self, frame, pre_frame):
        """Calculate the difference between frames."""
        raise NotImplementedError()

    @staticmethod
    def str_to_class(feature):
        feature_dict = {
            'edge': EdgeDiff,
            'pixel': PixelDiff,
            'area': AreaDiff,
            'corner': CornerDiff,
        }
        return feature_dict[feature]


class EdgeDiff(DiffProcessor):
    feature = 'edge'

    def get_frame_feature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        edge = cv2.Canny(blur, 50, 255)
        return edge

    def cal_frame_diff(self, edge, pre_edge):
        total_pixels = edge.shape[0] * edge.shape[1]
        frame_diff = cv2.absdiff(edge, pre_edge)
        frame_diff_binary = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)[1]
        changed_pixels = cv2.countNonZero(frame_diff_binary)
        fraction_changed = changed_pixels / total_pixels
        return fraction_changed


class PixelDiff(DiffProcessor):

    feature = 'pixel'

    def get_frame_feature(self, frame):
        return frame

    def cal_frame_diff(self, frame, pre_frame):
        total_pixels = frame.shape[0] * frame.shape[1]
        frame_diff = cv2.absdiff(frame, pre_frame)
        frame_diff_gray = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        frame_diff_binary = cv2.threshold(frame_diff_gray, 50, 255, cv2.THRESH_BINARY)[1]
        changed_pixels = cv2.countNonZero(frame_diff_binary)
        fraction_changed = changed_pixels / total_pixels
        return fraction_changed


class AreaDiff(DiffProcessor):

    feature = 'area'

    def get_frame_feature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        return blur

    def cal_frame_diff(self, frame, pre_frame):
        total_pixels = frame.shape[0] * frame.shape[1]
        frame_delta = cv2.absdiff(frame, pre_frame)
        frame_diff_binary = cv2.threshold(frame_delta, 50, 255, cv2.THRESH_BINARY)[1]
        dil = cv2.dilate(frame_diff_binary, None)
        contours = cv2.findContours(dil.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        if not contours:
            return 0.0
        return max([cv2.contourArea(c) / total_pixels for c in contours])


class CornerDiff(DiffProcessor):
    feature = 'corner'
    def get_frame_feature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corner = cv2.cornerHarris(gray, 5, 3, 0.05)
        corner = cv2.dilate(corner, None)
        return corner

    def cal_frame_diff(self, frame, pre_frame):
        total_pixels = frame.shape[0] * frame.shape[1]
        frame_diff = cv2.absdiff(frame, pre_frame)
        changed_pixels = cv2.countNonZero(frame_diff)
        fraction_changed = changed_pixels / total_pixels
        return fraction_changed