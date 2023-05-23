import cv2
import numpy as np
from loguru import logger
from mapcalc import calculate_map
from model_management.model_info import classes


def cal_iou(a, b):
    a_x1 = a[0]
    a_y1 = a[1]
    a_x2 = a[2]
    a_y2 = a[3]

    b_x1 = b[0]
    b_y1 = b[1]
    b_x2 = b[2]
    b_y2 = b[3]

    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection_area = inter_w * inter_h

    a_w = max(0, a_x2 - a_x1)
    a_h = max(0, a_y2 - a_y1)
    a_area = a_w * a_h

    b_w = max(0, b_x2 - b_x1)
    b_h = max(0, b_y2 - b_y1)
    b_area = b_w * b_h

    union_area = a_area + b_area - intersection_area
    return intersection_area / union_area


def get_offloading_region(high_detections, low_regions, img_shape):
    """
    Remove larger areas
    Remove overlapping areas
    :param detection:
    :param inferencer:
    :return:
    """
    offloading_region = []

    img_height = img_shape[0]
    img_width = img_shape[1]

    for i in range(len(low_regions)):
        x1 = low_regions[i][0]
        y1 = low_regions[i][1]
        x2 = low_regions[i][2]
        y2 = low_regions[i][3]
        if (x2 - x1) * (y2 - y1) > img_width * img_height * 0.1:
            continue
        if high_detections is not None:
            match = 0
            for j in range(len(high_detections)):
                if cal_iou(low_regions[i], high_detections[j]) > 0.3:
                    match += 1
            if match > 0:
                continue
        offloading_region.append(low_regions[i])
    return offloading_region


def get_offloading_image(offloading_region, image):
    """
    Put these areas on a black background and encode them with high quality
    :param region_query:
    :return:
    """
    cropped_image = np.zeros_like(image)
    width = image.shape[1]
    hight = image.shape[0]
    cached_image = image
    for i in range(len(offloading_region)):
        x1 = int(offloading_region[i][0])-1 if int(offloading_region[i][0])-1 >= 0 else 0
        y1 = int(offloading_region[i][1])-1 if int(offloading_region[i][1])-1 >= 0 else 0
        x2 = int(offloading_region[i][2])+1 if int(offloading_region[i][2])+1 <= width else width
        y2 = int(offloading_region[i][3])+1 if int(offloading_region[i][3])+1 <= hight else hight
        cropped_image[y1:y2, x1:x2, :] = cached_image[y1:y2, x1:x2, :]
    return cropped_image


def draw_detection(img, pred_boxes, pred_cls, pred_score):
    cached_img = img
    rect_th = 1
    text_th = 1
    text_size = 1
    if pred_boxes is None and pred_cls is None:
        return cached_img
    vehicle_classes = classes['vehicle']
    for i in range(len(pred_boxes)):
        # draw the image and label
        if pred_cls[i] in vehicle_classes:
            # Draw Rectangle with the coordinates
            cv2.rectangle(cached_img, (int(pred_boxes[i][0]), int(pred_boxes[i][1])), (int(pred_boxes[i][2]), int(pred_boxes[i][3])),
                          color=(0, 255, 0), thickness=rect_th)
            # Write the prediction class
            cv2.putText(cached_img, pred_cls[i], (int(pred_boxes[i][0]), int(pred_boxes[i][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    return cached_img


def cal_mAP(ground_truth, result_dict):
    # calculates the mAP for an IOU threshold of 0.5
    return calculate_map(ground_truth, result_dict, 0.5)

if __name__ == '__main__':
    pass