import time
import cv2
import argparse
import munch
import yaml

from edge.edge_worker import EdgeWorker
from tools.video_processor import VideoProcessor
from loguru import logger


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

    def get_result(self):
        result_dict = {
            'boxes': self.detection_boxes,
            'class': self.detection_class,
            'score': self.detection_score
        }
        return str(result_dict)

    def cal_accuracy(self):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="configuration description")
    parser.add_argument("--yaml_path", default="./config/config.yaml", help="input the path of *.yaml")
    args = parser.parse_args()
    with open(args.yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    #provide class-like access for dict
    config = munch.munchify(config)
    client_config = config.client
    edge = EdgeWorker(client_config)
    logger.add("log/client/client_{time}.log", level="INFO", rotation="500 MB")

    with VideoProcessor(config.video_path) as video:
        video_fps = video.fps
        index = 0
        dur = int(video_fps / client_config.fps)
        while True:
            print(index)
            frame = next(video)
            if frame is None:
                logger.info("the video finished")
                break
            index += 1
            if index % dur == 0:
                start_time = time.time()
                task = Task(index, frame, start_time)
                edge.frame_cache.put(task, block=True)
                time.sleep(1/client_config.fps)





