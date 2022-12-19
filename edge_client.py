import time
import cv2
import argparse
import munch
import yaml

from edge.edge_worker import EdgeWorker
from edge.task import Task
from tools.video_processor import VideoProcessor
from loguru import logger





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="configuration description")
    parser.add_argument("--yaml_path", default="./config/config.yaml", help="input the path of *.yaml")
    args = parser.parse_args()
    with open(args.yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    #provide class-like access for dict
    config = munch.munchify(config)
    client_config = config.client
    database_config = config.database
    client_config.update(database_config)
    edge = EdgeWorker(client_config)
    logger.add("log/client/client_{time}.log", level="INFO", rotation="500 MB")

    with VideoProcessor(config.video_path) as video:
        video_fps = video.fps
        index = 0
        dur = round(video_fps / client_config.fps)
        while True:
            frame = next(video)
            if frame is None:
                logger.info("the video finished")
                break
            index += 1
            if index % dur == 0:
                start_time = time.time()
                task = Task(client_config.edge_id, index, frame, start_time, frame.shape)
                edge.frame_cache.put(task, block=True)
                time.sleep(1/client_config.fps)





