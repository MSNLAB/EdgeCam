import argparse
import signal
import sys
import threading
import time


import wandb
import munch
import yaml

from edge.edge_worker import EdgeWorker
from loguru import logger

from edge.task import Task
from tools.file_op import clear_folder
from tools.video_processor import VideoProcessor

def signal_handler(signal, frame):
    logger.debug("Received Ctrl+C. Cleaning up...")
    clear_folder(config.retrain.cache_path)
    sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="configuration description")
    parser.add_argument("--yaml_path", default="./config/config.yaml", help="input the path of *.yaml")
    args = parser.parse_args()
    with open(args.yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    #provide class-like access for dict
    config = munch.munchify(config)
    wandb.init(project="filter", config=config)
    config = config.client
    signal.signal(signal.SIGINT, signal_handler)
    event = threading.Event()
    edge = EdgeWorker(config)
    logger.add("log/client/client_{time}.log", level="INFO", rotation="500 MB")
    try:
        with VideoProcessor(config.source) as video:
            video_fps = video.fps
            logger.info("the video fps is {}".format(video_fps))
            index = 0
            if config.interval == 0:
                logger.error("the interval error")
                sys.exit(1)
            logger.info("Take the frame interval is {}".format(config.interval))
            while True:
                frame = next(video)
                if frame is None:
                    logger.debug("The video finished")
                    break
                index += 1
                if index % config.interval == 0:
                    start_time = time.time()
                    task = Task(config.edge_id, index, frame, start_time, frame.shape)
                    edge.frame_cache.put(task, block=True)
                    time.sleep((config.interval * 1.0) / video_fps)
        event.wait()
    except KeyboardInterrupt:
        pass

