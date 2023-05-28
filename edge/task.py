from loguru import logger


class Task:
    def __init__(self, edge_id, frame_index, frame, start_time, raw_shape):
        self.edge_id = edge_id
        self.frame_index = frame_index
        self.frame_edge = frame
        self.start_time = start_time
        self.raw_shape = raw_shape
        self.state = None
        self.ref = None
        self.end_time = None
        self.frame_cloud = None
        self.other = False
        self.directly_cloud = False
        self.edge_process = False


        self.detection_boxes = []
        self.detection_class = []
        self.detection_score = []


    def add_result(self, detection_boxes, detection_class, detection_score):
        if detection_boxes is not None:
            assert len(detection_boxes) == len(detection_class) == len(detection_score)
            for i in range(len(detection_boxes)):
                self.detection_boxes.append(detection_boxes[i])
                self.detection_class.append(detection_class[i])
                self.detection_score.append(detection_score[i])

    def get_result(self):
        return self.detection_boxes, self.detection_class, self.detection_score
