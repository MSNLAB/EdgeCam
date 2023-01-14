


class Task:
    def __init__(self, edge_id, frame_index, frame, start_time, raw_shape):
        self.edge_id = edge_id
        self.frame_index = frame_index
        self.frame_edge = frame
        self.start_time = start_time
        self.raw_shape = raw_shape
        self.end_time = None
        self.frame_cloud = None
        self.other = False
        self.directly_cloud = False
        self.edge_process = False
        self.detection_boxes = []
        self.detection_class = []
        self.detection_score = []

    def set_frame_cloud(self, offloading_frame):
        self.frame_cloud = offloading_frame

    def set_end_time(self, end_time):
        self.end_time = end_time

    def add_result(self, detection_boxes, detection_class, detection_score):
        assert len(detection_boxes) == len(detection_class) == len(detection_score)
        for i in range(len(detection_boxes)):
            self.detection_boxes.append(detection_boxes[i])
            self.detection_class.append(detection_class[i])
            self.detection_score.append(detection_score[i])

    def get_result(self):
        result_dict = {
            'boxes': self.detection_boxes,
            'class': self.detection_class,
            'score': self.detection_score
        }
        return str(result_dict)
