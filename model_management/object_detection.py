import cv2
import torch
import os
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader
from torchvision.models.detection.backbone_utils import*
from model_management.detection_dataset import TrafficDataset
from model_management.detection_metric import RetrainMetric
from model_management.model_info import model_lib, COCO_INSTANCE_CATEGORY_NAMES, classes
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import *

from model_management.utils import get_offloading_region, get_offloading_image

def _collate_fn(batch):
    return tuple(zip(*batch))

class Object_Detection:
    def __init__(self, config, type):
        self.type = type
        self.config = config
        self.init_model_flag = False
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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.model.eval()

    def init_model(self):
        logger.debug("init_model")
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.roi_heads.parameters():
            logger.debug("roi")
            param.requires_grad = True

        for module in self.model.roi_heads.modules():
            logger.debug("{}".format(module))
            if isinstance(module, nn.Conv2d):
                logger.debug("conv")
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                logger.debug("batch")
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        torch.save(self.model.state_dict(), "./model_management/tmp_model.pth")

    def retrain(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tmp_model = eval(self.model_name)(pretrained_backbone=False, pretrained=False)
        state_dict = torch.load("./model_management/tmp_model.pth", map_location=device)
        tmp_model.load_state_dict(state_dict)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.roi_heads.parameters():
            param.requires_grad = True

        dataset = TrafficDataset(root="./retrain_data")
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
                print("pre tar {}".format(targets))
                for t in targets:
                    print("t {}".format(t))
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                print(targets)
                with torch.cuda.amp.autocast():
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
        self.model.load_state_dict(state_dict)
        self.model.eval()


    def small_inference(self, img):
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img = img.to(device)
        #get the inference result
        res = self.model([img])
        logger.debug("res {}".format(res))
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






