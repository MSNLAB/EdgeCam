import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from inferencer.detection_transforms import Compose, ToTensor, Resize


class DetectionDataset(Dataset):
    def __init__(self, frames, transforms=None):
        if transforms is None:
            transforms = Compose((
                ToTensor(),
            ))
        self.frames = frames
        self.transforms = transforms
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame_info = self.frames[index]
        img = Image.open(frame_info['path']).convert("RGB")
        target = {"boxes": torch.as_tensor(frame_info['boxes'], dtype=torch.float32),
                  "labels": torch.as_tensor(frame_info['labels'], dtype=torch.int64),}
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        print("img {} \n target {}".format(img, target))
        return img, target


class TrafficDataset(DetectionDataset):
    def __init__(self, root=None):
        self.root = root
        frames = collect_frames(root)
        super(TrafficDataset, self).__init__(frames)



annotation_cols = ('frame_index', 'target_id', 'bbox_x1', 'bbox_y1',
                   'bbox_x2', 'bbox_y2', 'score', 'object_category',)

def collect_frames(root):
    frames = []
    frame_path = os.path.join(root, 'frames')
    frame_names = list(os.listdir(frame_path))

    annotation_path = os.path.join(root, 'annotations.txt')
    annotations = pd.read_csv(annotation_path, header=None, names=annotation_cols)

    for frame_name in frame_names:
        _id = int(frame_name.split('.')[0])
        _path = os.path.join(frame_path, frame_name)
        _labels = annotations[annotations['frame_index'] == _id]

        boxes = []
        labels = []
        for _idx, _label in _labels.iterrows():
            label = _label['target_id']
            if label != 0:
                x_min = _label['bbox_x1']
                y_min = _label['bbox_y1']
                x_max = _label['bbox_x2']
                y_max = _label['bbox_y2']
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(label)

        if len(boxes) != 0 and len(labels) != 0:
            frames.append({'path': _path, 'frame_index': _id,
                           'boxes': boxes, 'labels': labels})

    return frames





