import cv2
import torch
import os
import numpy as np
from inferencer.model_info import model_lib, COCO_INSTANCE_CATEGORY_NAMES, classes
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import *

from inferencer.utils import get_offloading_region, get_offloading_image

"""
1. 高置信度区域并且目标是vehicle认为是正确的。
2. 低置信度区域中：去掉占据面积较大的（认为可以很好识别），
去掉和高置信度区域重合超过30%的区域（认为是相同区域），
得到需要高质量编码的区域。
3. 对两部分进行分别编码，然后合并为一张图像
4. 传输到大模型上，进行检测。
5. 最后将大模型得到的结果与之前的结果进行合并
"""

class Object_Detection:
    def __init__(self, config, type):
        self.type = type
        if type == 'small inference':
            self.model_name = config.small_model_name
        else:
            self.model_name = config.large_model_name
        self.model = self.load_model()
        self.threshold_low = 0.5
        self.threshold_high = 0.8

    def load_model(self):
        weight_folder = os.path.join(os.path.dirname(__file__), 'models')
        if self.model_name in model_lib.keys():
            weight_files_path = \
                os.path.join(weight_folder, model_lib[self.model_name]['model_path'])
            weight_load = torch.load(weight_files_path)
            model = eval(self.model_name)(pretrained_backbone=False, pretrained=False)
            model.load_state_dict(weight_load)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            return model
        else:
            return None

    def small_inference(self, img):
        pred_boxes, pred_class, pred_score = self.get_model_prediction(img, self.threshold_low)
        if pred_boxes == None:
            return 'object not detected'
        #filter high confidence region as the detection result
        prediction_index = [pred_score.index(x) for x in pred_score if x > self.threshold_high][-1]
        detection_boxes = pred_boxes[:prediction_index + 1]
        detection_class = pred_class[:prediction_index + 1]
        detection_score = pred_score[:prediction_index + 1]
        #split into high and low confidence region
        high_detections = detection_boxes
        low_regions = pred_boxes
        #get the inferencer that need to query
        offloading_region = get_offloading_region(high_detections, low_regions, img.shape)
        if len(offloading_region) == 0:
            offloading_image = None
        else:
            offloading_image = get_offloading_image(offloading_region, img)

        return offloading_image, detection_boxes, detection_class, detection_score


    def large_inference(self, img):
        pred_boxes, pred_class, pred_score = self.get_model_prediction(img, self.threshold_high)
        return pred_boxes, pred_class, pred_score

    def get_model_prediction(self, img, threshold):
        #process the image
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img = img.to(device)
        #get the inference result
        res = self.model([img])

        if torch.cuda.is_available():
            prediction_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(res[0]['labels'].cuda().data.cpu().numpy())]
            prediction_boxes = [[i[0], i[1], i[2], i[3]] for i in list(res[0]['boxes'].detach().cpu().numpy())]
            prediction_score = list(res[0]['scores'].detach().cpu().numpy())

        else:
            prediction_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(res[0]['labels'].numpy())]
            prediction_boxes = [[i[0], i[1], i[2], i[3]] for i in list(res[0]['boxes'].detach().numpy())]
            prediction_score = list(res[0]['scores'].detach().numpy())

        try:
            prediction_t = [prediction_score.index(x) for x in prediction_score if x > threshold][-1]
        except IndexError:
            return None, None, None
        else:
            pred_boxes = prediction_boxes[:prediction_t+1]
            pred_class = prediction_class[:prediction_t+1]
            pred_score = prediction_score[:prediction_t+1]

            detections_vehicle_boxes = []
            detections_vehicle_class = []
            detections_vehicle_score = []
            vehicle_classes = classes['vehicle']
            for i in range(len(pred_boxes)):
                if pred_class[i] in vehicle_classes:
                    detections_vehicle_boxes.append(pred_boxes[i])
                    detections_vehicle_class.append(pred_class[i])
                    detections_vehicle_score.append(pred_score[i])
            if len(detections_vehicle_boxes) == 0:
                return None, None, None
            return detections_vehicle_boxes, detections_vehicle_class, detections_vehicle_score





