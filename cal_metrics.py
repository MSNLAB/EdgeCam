import argparse
import json
import munch
import numpy as np
import yaml
from loguru import logger
from mapcalc import calculate_map
from datetime import datetime
from database.database import DataBase
from inferencer.object_detection import Object_Detection
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
        self.config = config
        self.source = config.source
        self.client_conifg = config.client
        self.inference_config = config.server
        self.database_config = config.database
        self.large_object_detection = Object_Detection(self.inference_config, type='large inference')

    def cal_ground_truth(self):
        with VideoProcessor(self.source) as video:
            truth_dict = {}
            video_fps = video.fps
            interval = self.client_conifg.interval
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
        with open('truth.json', 'r') as f:
            ground_truths = json.load(f)
        # calculates the mAP for an IOU threshold of 0.5
        database = DataBase(self.database_config)
        database.use_database()
        result = database.select_result(self.client_conifg.edge_id)
        total_frame = 180
        edge_drop_count = 0
        cloud_drop_count = 0
        sum_map = 0.0
        map_rm_direct_drop = 0.0

        first = True
        for (index, start_time, end_time, res, note) in result:
            if first:
                result_dict = {'boxes': [], 'labels': [], 'scores': []}
                start = start_time
                all_delay = start_time-start_time
                logger.debug(all_delay)
                first = False
            end = end_time
            gap = end_time-start_time
            logger.debug(gap)
            all_delay += gap
            if res != "":
                result_dict = eval(res)
            else:
                logger.debug("index {}, reuse{} ".format(str(index), result_dict))

            logger.debug("res {}".format(result_dict))
            ground_truth = ground_truths['{}'.format(index)]
            logger.debug("truth {}".format(ground_truth))
            cal_map = calculate_map(ground_truth, result_dict, 0.5)
            sum_map += cal_map
            if note == "Timeout":
                if res == "":
                    edge_drop_count += 1
                else:
                    cloud_drop_count += 1
            if res != "":
                map_rm_direct_drop += cal_map

            logger.debug(cal_map)

        sum_delay = end - start

        print("edge, cloud, total drop number: {} {} {}".format(
            edge_drop_count, cloud_drop_count, edge_drop_count+cloud_drop_count))
        print("edge, cloud, total drop rate {},{},{}".format(
            edge_drop_count/total_frame, cloud_drop_count/total_frame, (edge_drop_count+cloud_drop_count)/total_frame))
        print("the map sum {}".format(sum_map,))
        print("total average map {}".format(
            sum_map/total_frame))
        print("sum delay, throughput delay {} {}, and average delay {} {}".format(sum_delay, all_delay,
                                                    sum_delay/total_frame, all_delay/total_frame))

    def cal_mAP_with_diff(self):
        interval = self.client_conifg.interval
        with open('truth.json', 'r') as f:
            ground_truths = json.load(f)
        # calculates the mAP for an IOU threshold of 0.5
        database = DataBase(self.database_config)
        database.use_database()
        result = database.select_result(self.client_conifg.edge_id)
        total_frame = 90
        edge_drop_count = 0
        cloud_drop_count = 0
        sum_map = 0.0
        map_rm_direct_drop = 0.0
        first = True
        i = 0

        while i < len(result):
            index, start_time, end_time, res, note = result[i]
            if first:
                all_delay = start_time - start_time
                start = start_time
                first = False
                result_dict = {'boxes': [], 'labels': [], 'scores': []}
            gap = end_time-start_time
            all_delay += gap
            end = end_time
            # not timeout
            if res != "":
                result_dict = eval(res)
            while i+1 < len(result) and index < int(result[i+1][0]):

                logger.debug("index {}, res {}".format(str(index), result_dict))
                ground_truth = ground_truths['{}'.format(index)]
                logger.debug("truth {}".format(ground_truth))
                cal_map = calculate_map(ground_truth, result_dict, 0.5)
                sum_map += cal_map
                index += interval
                logger.debug(cal_map)
            if note == "Timeout":
                if res == "":
                    edge_drop_count += 1
                else:
                    cloud_drop_count += 1
            i += 1
        sum_delay = end - start

        print("the unfiltered count/rate {} {}".format(len(result), len(result)/total_frame))
        print("edge, cloud, total drop count {} {} {}".format(
            edge_drop_count, cloud_drop_count, edge_drop_count + cloud_drop_count))
        print("edge, cloud, total drop rate {},{},{}".format(
            edge_drop_count / total_frame, cloud_drop_count / total_frame,
            (edge_drop_count + cloud_drop_count) / total_frame))
        print("sum map {}".format(sum_map,))
        print("total average map {},".format(
            sum_map / total_frame,))
        print("sum delay, throughput delay {} {}, and average delay {} {}".format(
            sum_delay, all_delay, sum_delay/total_frame, all_delay/len(result)))

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
    if config.client.diff_lable:
        cal.cal_mAP_with_diff()
    else:
        cal.cal_mAP()



