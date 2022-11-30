import json
import numpy as np
from scipy.io._idl import AttrDict

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



class Ground_truth:
    def __init__(self, config):
        self.video_path = config.source_path
        self.large_object_detection = Object_Detection(config, type='large inference')

    def cal_ground_truth(self):
        with VideoProcessor(config.source_path) as video:
            truth_dict = {}
            video_fps = video.fps
            index = 0
            dur = int(video_fps / config.fps)
            while True:
                print(index)
                frame = next(video)
                if frame is None:
                    print("the video is over")
                    break
                if index % dur == 0:
                    truth_boxes, truth_class, _ = self.large_object_detection.large_inference(frame)
                    truth_dict['{}'.format(index)] = {
                        'labels': truth_class,
                        'boxes': truth_boxes,
                    }
                    index += 1
                if index > 1:
                    break
        with open('truth.json', 'w') as f:
            json.dump(truth_dict, f, indent=4, cls=Encoder)





if __name__ == '__main__':
    config = AttrDict()
    config.fps = 30
    config.source_path = '../video_data/road.mp4'
    config.large_model_name = 'fasterrcnn_resnet50_fpn'
    #Ground_truth(config).cal_ground_truth()
    with open('truth.json', 'r') as f:
        truth = json.load(f)


