import json
import munch
import numpy as np
from loguru import logger
import sys
sys.path.append("../")
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


class GroundTruth:
    def __init__(self, config):
        self.config = config
        self.source = config.source
        #self.large_object_detection = Object_Detection(config, type='large inference')
        self.small_object_detection = Object_Detection(config, type='small inference')

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
                    print("the video is over")
                    break
                index += 1
                logger.info("the frame index is {}".format(index))
                if index % interval == 0:
                    #truth_boxes, truth_class, _ = self.large_object_detection.large_inference(frame)
                    _, truth_boxes, truth_class, _ = self.small_object_detection.small_inference(frame)
                    truth_dict['{}'.format(index)] = {
                        'labels': truth_class,
                        'boxes': truth_boxes,
                    }
                if index > 4:
                    break
        with open('truth.json', 'w') as f:
            json.dump(truth_dict, f, indent=4, cls=Encoder)





if __name__ == '__main__':
    config = {
        'source':
            {
                'video_path': '../video_data/road.mp4',
                'rtsp': None,
            },
        'interval': 1,
        'large_model_name': 'fasterrcnn_resnet50_fpn',
        'small_model_name': 'fasterrcnn_mobilenet_v3_large_320_fpn',
        }

    config = munch.munchify(config)
    GroundTruth(config).cal_ground_truth()
    #with open('truth.json', 'r') as f:
    #    truth = json.load(f)


