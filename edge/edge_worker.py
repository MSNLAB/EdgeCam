import threading
import time
from datetime import datetime

import cv2
import grpc
import numpy as np
from queue import Queue
from loguru import logger

from database.database import DataBase
from difference.diff import EdgeDiff
from grpc_server import message_transfer_pb2_grpc, message_transfer_pb2
from tools.convert_tool import cv2_to_base64
from tools.preprocess import frame_resize
from inferencer.object_detection import Object_Detection


class offloading_policy:
    def __init__(self, policy_name, local_queue, offloading_queue):
        self.policy_name = policy_name
        self.local_queue = local_queue
        self.offloading_queue = offloading_queue

    def threshold_offloading_policy(self, task):
        if self.local_queue.qsize() <= 3:
            # put into local queue
            self.local_queue.put(task, block=True)
        else:
            task.set_offloading_frame(task.frame)
            self.offloading_queue.put(task, block=True)

    def get_policy_method(self):
        policy_dict = {
            "threshold_offloading_policy": self.threshold_offloading_policy,
        }
        if self.policy_name not in policy_dict:
            logger.error("the policy name is error")
        return policy_dict[self.policy_name]



class EdgeWorker:
    def __init__(self, config):

        self.config = config
        self.server_url = config.server_url

        self.edge_processor = EdgeDiff()
        self.small_object_detection = Object_Detection(config, type='small inference')

        #create database and tables
        self.database = DataBase(config.database)
        self.database.use_database()
        self.database.create_tables()

        self.frame_cache = Queue(config.frame_cache_maxsize)
        self.local_queue = Queue(config.local_queue_maxsize)
        self.offloading_queue = Queue(config.offloading_queue_maxsize)
        #task offload policy
        self.policy = offloading_policy(config.policy, self.local_queue, self.offloading_queue)
        self.task_decision = self.policy.get_policy_method()
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
        diff = 1.0
        while True:
            #get task from cache
            task = self.frame_cache.get(block=True)
            frame = task.frame
            if diff == 1.0:
                pre_frame_feature = self.edge_processor.get_frame_feature(frame)
                self.local_queue.put(task, block=True)
                task = self.frame_cache.get(block=True)
                frame = task.frame
                diff = 0.0
            frame_feature = self.edge_processor.get_frame_feature(frame)
            # calculate and accumulate the difference
            diff += self.edge_processor.cal_frame_diff(frame_feature, pre_frame_feature)
            pre_frame_feature = frame_feature
            if diff > self.config.diff_thresh:
                diff = 0.0
                self.task_decision(task)

    def local_worker(self):
        while True:
            #get a inference task from local queue
            current_task = self.local_queue.get(block=True)
            current_frame = current_task.frame
            #get the query image and the small inference result
            samll_result = self.small_object_detection.small_inference(current_frame)

            #put the cropped image into offloading queue
            if samll_result != 'object not detected':
                offloading_image, detection_boxes, detection_class, detection_score = samll_result
                if offloading_image is not None:
                    current_task.set_offloading_frame(offloading_image)
                    current_task.add_result(detection_boxes, detection_class, detection_score)
                    self.offloading_queue.put(current_task, block=True)
                else:
                    end_time = time.time()
                    current_task.set_end_time(end_time)
                    current_task.add_result(detection_boxes, detection_class, detection_score)
                    #
                    str_result = current_task.get_result()
                    in_data = (
                        current_task.task_index,
                        datetime.fromtimestamp(current_task.start_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
                        datetime.fromtimestamp(current_task.end_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
                        str_result,)
                    print(in_data)
                    self.database.insert_data(table_name='result', data=in_data)
            else:
                logger.info('This frame has no objects detected')

    def offload_worker(self):
        while True:
            #get cropped image
            current_task = self.offloading_queue.get(block=True)
            cropped_image = current_task.offloading_frame
            #preprocess
            changed_image = frame_resize(cropped_image, self.config.new_height)
            #scaling radio
            scale = cropped_image.shape[0] / self.config.new_height

            #encoded the cropped_image
            encoded_image = cv2_to_base64(changed_image, qp=self.config.quality)
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
                current_task.set_end_time(end_time)
                if detection_boxes is not None:
                    #Convert the result corresponding to the original resolution
                    detection_boxes = (np.array(detection_boxes) * scale).tolist()
                    current_task.add_result(detection_boxes, detection_class, detection_score)
                #
                str_result = current_task.get_result()
                in_data = (
                    current_task.task_index,
                    datetime.fromtimestamp(current_task.start_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
                    datetime.fromtimestamp(current_task.end_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
                    str_result,)
                print(in_data)
                self.database.insert_data(table_name='result', data=in_data)