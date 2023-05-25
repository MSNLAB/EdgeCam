model_lib = {
    'fasterrcnn_resnet50_fpn': {
        'model_path': 'fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
    },
    'fasterrcnn_mobilenet_v3_large_fpn': {
        'model_path': 'fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth',
    },
    'fasterrcnn_mobilenet_v3_large_320_fpn': {
        'model_path': 'fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth',}
}


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

classes = {
    "vehicle": ['car', 'bus', 'train', 'truck'],
    "persons": [1, 2, 4],
    "roadside-objects": [10, 11, 13, 14]
}


annotation_cols = ('frame_index', 'target_id', 'bbox_x1', 'bbox_y1',
                   'bbox_x2', 'bbox_y2', 'score', 'object_category',)