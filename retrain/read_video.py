import os

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

_to_tensor = T.Compose([T.ToTensor(), ])


def video_reader(video_path, frame_path, label_model=None, annotation_path=None, device=None):
    video_capture = cv2.VideoCapture(video_path)
    success, frame = video_capture.read()
    index = 0
    interval = 1
    annotations = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    while success:
        cv2.imwrite(
            os.path.join(frame_path, str(index) + '.jpg'),
            frame
        )
        if label_model is not None:
            label_model.to(device)
            label_model.eval()
            with torch.no_grad():
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img = _to_tensor(img)
                img = img.to(device)
                pred = label_model([img])
                print("img {} \n pred {}".format(img, pred))
                for score, label, box in zip(pred[0]['scores'], pred[0]['labels'], pred[0]['boxes']):
                    score = score.cpu().detach().item()
                    label = label.cpu().detach().item()
                    box = box.cpu().detach().tolist()
                    if score >= 0.20:
                        annotations.append(
                            (index, label, box[0], box[1],
                            box[2], box[3],score)
                        )

        success, frame = video_capture.read()
        index += 1
        if not success:
            break
        if index > 1:
            break

    if len(annotations):
        np.savetxt(annotation_path, annotations, fmt=['%d','%d','%f','%f','%f','%f','%f'], delimiter=',')


