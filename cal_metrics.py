import argparse
import json
import munch
import numpy as np
import yaml
from loguru import logger
from mapcalc import calculate_map
from datetime import datetime
from database.database import DataBase
from model_management.object_detection import Object_Detection
from tools.preprocess import frame_resize
from tools.video_processor import VideoProcessor

class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(Encoder, self).default(obj)


class Cal_Metrics:
    def __init__(self, config):
        self.config = config.client
        self.source = config.client.source
        self.database_config = config.client.database
        self.inference_config = config.server
        self.large_object_detection = Object_Detection(self.inference_config, type='large inference')

    def cal_ground_truth(self):
        with VideoProcessor(self.source) as video:
            truth_dict = {}
            video_fps = video.fps
            interval = self.config.interval
            logger.info("the video fps is {}".format(video_fps))
            if interval == 0:
                logger.error("the interval error")
            logger.info("Take the frame interval is {}".format(interval))
            index = 0
            while True:
                frame = next(video)
                if frame is None:
                    logger.info("the video is over")
                    break
                index += 1
                logger.info("the frame index is {}".format(index))
                if index % interval == 0:
                    truth_boxes, truth_class, _ = self.large_object_detection.large_inference(frame)
                    truth_dict['{}'.format(index)] = {
                        'labels': truth_class,
                        'boxes': truth_boxes,
                    }
        with open('truth.json', 'w') as f:
            json.dump(truth_dict, f, indent=4, cls=Encoder)


    def cal_mAP(self):
        interval = self.config.interval
        with open('truth.json', 'r') as f:
            ground_truths = json.load(f)
        # calculates the mAP for an IOU threshold of 0.5
        database = DataBase(self.database_config)
        database.use_database()
        result = database.select_result(self.config.edge_id)
        total_frame = self.source.max_count / interval
        sum_map = 0.0
        sum_delay = 0.0
        filtered_out = 0
        i = 0
        last_result = None
        logger.debug(len(result))
        while i < len(result):
            index, start_time, end_time, res, log = result[i]
            gap = end_time-start_time
            sum_delay += gap
            result_dict = eval(res)
            logger.debug(result_dict.keys())
            if 'lables' in result_dict.keys():
                pred = eval(res)
                last_result = pred
            elif 'ref' in result_dict.keys():
                filtered_out += 1
                pred = last_result
            else:
                pred = last_result
            logger.debug(pred)
            ground_truth = ground_truths['{}'.format(index)]
            logger.debug(ground_truth)
            map = calculate_map(ground_truth, pred, 0.5)
            sum_map += map



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="configuration description")
    parser.add_argument("--yaml_path", default="./config/config.yaml", help="input the path of *.yaml")
    args = parser.parse_args()
    with open(args.yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    # provide class-like access for dict
    config = munch.munchify(config)
    cal_truth = False
    cal = Cal_Metrics(config)
    if cal_truth:
        cal.cal_ground_truth()
    cal.cal_mAP()



