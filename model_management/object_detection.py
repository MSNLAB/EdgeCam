import threading

import cv2
import pandas as pd
import torch
import os
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader
from torchvision.models.detection.backbone_utils import*
from model_management.detection_dataset import TrafficDataset
from model_management.detection_metric import RetrainMetric
from model_management.model_info import model_lib, COCO_INSTANCE_CATEGORY_NAMES, classes, annotation_cols
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import *
from mapcalc import calculate_map
from model_management.utils import get_offloading_region, get_offloading_image

def _collate_fn(batch):
    return tuple(zip(*batch))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.debug(device)

class Object_Detection:
    def __init__(self, config, type):
        self.type = type
        self.config = config
        self.init_model_flag = False
        self.model_lock = threading.Lock()
        if type == 'small inference':
            self.model_name = config.small_model_name
            self.init_model_flag = True
        else:
            self.model_name = config.large_model_name
        self.model = None
        self.load_model()
        self.threshold_low = 0.2
        self.threshold_high = 0.6

    def load_model(self):
        weight_folder = os.path.join(os.path.dirname(__file__), 'models')
        if self.model_name in model_lib.keys():
            weight_files_path = \
                os.path.join(weight_folder, model_lib[self.model_name]['model_path'])
            weight_load = torch.load(weight_files_path)
            self.model = eval(self.model_name)(pretrained_backbone=False, pretrained=False)
            self.model.load_state_dict(weight_load)
            if self.init_model_flag:
                self.init_model()

            self.model.to(device)
            self.model.eval()

    def init_model(self):
        logger.debug("init_model")
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.roi_heads.parameters():
            param.requires_grad = True

        for module in self.model.roi_heads.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
        torch.save(self.model.state_dict(), "./model_management/tmp_model.pth")

    def retrain(self, path, select_index):

        tmp_model = eval(self.model_name)(pretrained_backbone=False, pretrained=False)
        state_dict = torch.load("./model_management/tmp_model.pth", map_location=device)
        tmp_model.load_state_dict(state_dict)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.roi_heads.parameters():
            param.requires_grad = True

        dataset = TrafficDataset(root=path, select_index = select_index)
        data_loader = DataLoader(dataset=dataset, batch_size=2, collate_fn=_collate_fn, )
        tr_metric = RetrainMetric()

        # 训练设置
        num_epoch = self.config.retrain.num_epoch
        roi_parameters = [p for p in tmp_model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(roi_parameters, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        for epoch in range(num_epoch):
            tmp_model.train()
            for images, targets in tr_metric.log_iter(epoch, num_epoch, data_loader):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = tmp_model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                tr_metric.update(loss_dict, losses)
            # Update the learning rate
            lr_scheduler.step()
        torch.save(tmp_model.state_dict(), "./model_management/tmp_model.pth")
        state_dict = torch.load("./model_management/tmp_model.pth", map_location=device)
        with self.model_lock:
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def model_evaluation(self,cache_path, select_index):
        map = []
        frame_path = os.path.join(cache_path, 'frames')
        annotation_path = os.path.join(cache_path, 'annotation.txt')
        annotations_f = pd.read_csv(annotation_path, header=None, names=annotation_cols)
        for _id in select_index:
            logger.debug(_id)
            path = os.path.join(frame_path, str(_id)+'.jpg')
            frame = cv2.imread(path)
            pred_boxes, pred_class, pred_score = self.get_model_prediction(frame, self.threshold_high)
            pred = {'labels':pred_class, 'boxes': pred_boxes, 'scores':pred_score}
            annos = annotations_f[annotations_f['frame_index'] == _id]
            target_boxes = []
            target_labels = []
            for _idx, _label in annos.iterrows():
                label = _label['target_id']
                if label != 0:
                    x_min = _label['bbox_x1']
                    y_min = _label['bbox_y1']
                    x_max = _label['bbox_x2']
                    y_max = _label['bbox_y2']
                    target_boxes.append([x_min, y_min, x_max, y_max])
                    target_labels.append(label)
            target = {'labels':target_labels, 'boxes': target_boxes}
            cal_map = calculate_map(target, pred, 0.5)
            logger.debug(cal_map)
            map.append(cal_map)
        map = np.mean(map)
        logger.debug(map)

    def small_inference(self, img):
        with self.model_lock:
            pred_boxes, pred_class, pred_score = self.get_model_prediction(img, self.threshold_low)
        if pred_boxes == None:
            return None, None, None, None
        #filter high confidence region as the detection result
        try:
            prediction_index = [pred_score.index(x) for x in pred_score if x > self.threshold_high][-1]
        except IndexError:
            detection_boxes = None
            detection_class = None
            detection_score = None
        else:
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
        img = img.to(device)
        #get the inference result
        res = self.model([img])
        if torch.cuda.is_available():
            prediction_class = list(res[0]['labels'].cuda().data.cpu().numpy())
            prediction_boxes = [[i[0], i[1], i[2], i[3]] for i in list(res[0]['boxes'].detach().cpu().numpy())]
            prediction_score = list(res[0]['scores'].detach().cpu().numpy())

        else:
            prediction_class = list(res[0]['labels'].numpy())
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
            return pred_boxes, pred_class, pred_score






