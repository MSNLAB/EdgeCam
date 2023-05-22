from typing import List, Tuple, Dict, Optional
import torch
import torchvision
from torch import nn, Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(nn.Module):
    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.to_tensor(image)
        return image, target


class Resize(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        width, height = F._get_image_size(image)
        image = F.resize(image, self.size)
        if target is not None:
            scale_w = self.size[1] / width
            scale_h = self.size[0] / height
            target["boxes"][:, [0, 2]] *= scale_w
            target["boxes"][:, [1, 3]] *= scale_h
        return image, target

