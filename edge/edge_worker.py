import threading
import time

import cv2
import grpc
from queue import Queue

from difference.diff import EdgeDiff
from grpc_server import message_transfer_pb2_grpc, message_transfer_pb2
from tools.convert_tool import cv2_to_base64
from tools.preprocess import frame_resize
from inferencer.object_detection import Object_Detection


class EdgeWorker:
    def __init__(self, config):

        self.config = config
        self.diff = 1.0
        self.policy = 'common'
        self.server_url = 'localhost:50051'

        self.edge_processor = EdgeDiff()
        self.small_object_detection = Object_Detection(config, type='small inference')

        self.frame_cache = Queue(config.frame_cache_maxsize)
        self.local_queue = Queue(config.local_queue_maxsize)
        self.offloading_queue = Queue(config.offloading_queue_maxsize)
        # start the thread for diff
        self.diff_processor = threading.Thread(target=self.diff_worker,)
        self.diff_processor.start()
        # start the thread for local process
        self.local_processor = threading.Thread(target=self.local_worker,)
        self.local_processor.start()
        # start the thread for offload
        self.offload_processor = threading.Thread(target=self.offload_worker,)
        self.offload_processor.start()

    def diff_worker(self):
        while True:
            #get task from cache
            task = self.frame_cache.get(block=True)
            frame = task.frame
            if self.diff == 1.0:
                pre_frame_feature = self.edge_processor.get_frame_feature(frame)
                self.local_queue.put(task, block=True)
                task = self.frame_cache.get(block=True)
                frame = task.frame
                self.diff = 0.0
            frame_feature = self.edge_processor.get_frame_feature(frame)
            # calculate and accumulate the difference
            self.diff += self.edge_processor.cal_frame_diff(frame_feature, pre_frame_feature)
            pre_frame_feature = frame_feature
            if self.diff > self.config.diff_thresh:
                self.diff = 0.0
                if self.policy == 'common':
                    # put into local queue
                    self.local_queue.put(task, block=True)
                else:
                    print('directly')
                    task.set_offloading_frame = frame
                    self.offloading_queue.put(task, block=True)

    def local_worker(self):
        while True:
            #get a inference task from local queue
            current_task = self.local_queue.get(block=True)
            current_frame = current_task.frame
            #get the query image and the small inference result
            flag, cropped_image, detection_boxes, detection_class, detection_score = \
                self.small_object_detection.small_inference(current_frame)
            print(flag)

            #put the cropped image into offloading queue
            if flag == 'offloading':
                current_task.set_offloading_frame(cropped_image)
                current_task.add_result(detection_boxes, detection_class, detection_score)
                self.offloading_queue.put(current_task, block=True)
            else:
                end_time = time.time()
                current_task.set_end_time(end_time)
                current_task.add_result(detection_boxes, detection_class, detection_score)
                #
                current_task.cal_accuracy()

    def offload_worker(self):
        while True:
            #get cropped image
            task = self.offloading_queue.get(block=True)
            cropped_image = task.offloading_frame
            #preprocess
            changed_image = frame_resize(cropped_image, 1080)
            #encoded the cropped_image
            encoded_image = cv2_to_base64(changed_image, qp=100)
            #covert to dict
            info = {
                'frame': encoded_image,
                'frame_shape': str(changed_image.shape)
            }
            with grpc.insecure_channel(self.server_url) as channel:
                stub = message_transfer_pb2_grpc.MessageTransferStub(channel)
                request = message_transfer_pb2.MessageRequest(frame=info['frame'], frame_shape=info['frame_shape'])
                response = stub.image_processor(request)
                result_dict = eval(response.result)
                detection_boxes = result_dict['boxes']
                detection_class = result_dict['class']
                detection_score = result_dict['score']

                end_time = time.time()
                task.set_end_time(end_time)
                if detection_boxes is not None:
                    task.add_result(detection_boxes, detection_class, detection_score)
                #
                task.cal_accuracy()