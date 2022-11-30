import time

import cv2
from scipy.io._idl import AttrDict
from edge.edge_worker import EdgeWorker
from tools.video_processor import VideoProcessor

class Task:
    def __init__(self, task_index, frame, start_time):
        self.task_index = task_index
        self.frame = frame
        self.start_time = start_time
        self.end_time = 0
        self.accuracy = 0
        self.offloading_frame = None
        self.detection_boxes = []
        self.detection_class = []
        self.detection_score = []

    def set_offloading_frame(self, offloading_frame):
        self.offloading_frame = offloading_frame

    def set_end_time(self, end_time):
        self.end_time = end_time

    def add_result(self, detection_boxes, detection_class, detection_score):
        self.detection_boxes.append(detection_boxes)
        self.detection_class.append(detection_class)
        self.detection_score.append(detection_score)

    def cal_accuracy(self):
        pass


if __name__ == '__main__':
    config = AttrDict()
    config.source_path = './video_data/road.mp4'
    config.diff_thresh = 0.005
    config.local_queue_maxsize = 10
    config.offloading_queue_maxsize = 10
    config.frame_cache_maxsize = 100
    config.samll_model_name = 'fasterrcnn_mobilenet_v3_large_fpn'
    config.large_model_name = 'fasterrcnn_resnet50_fpn'
    config.fps = 30

    edge = EdgeWorker(config)

    with VideoProcessor(config.source_path) as video:
        video_fps = video.fps
        index = 0
        dur = int(video_fps / config.fps)
        while True:
            print(index)
            frame = next(video)
            if frame is None:
                print("the video is over")
                break
            index += 1
            if index % dur == 0:
                start_time = time.time()
                task = Task(index, frame, start_time)
                edge.frame_cache.put(task, block=True)





