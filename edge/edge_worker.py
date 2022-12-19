import threading
import time
from concurrent import futures
from datetime import datetime

import random
import grpc
import numpy as np
from queue import Queue
from loguru import logger

from database.database import DataBase
from difference.diff import DiffProcessor
from grpc_server import message_transmission_pb2_grpc, message_transmission_pb2
from rpc_server import MessageTransmissionServicer

from tools.convert_tool import cv2_to_base64
from tools.preprocess import frame_resize
from inferencer.object_detection import Object_Detection




class EdgeWorker:
    def __init__(self, config):

        self.config = config
        self.edge_id = config.edge_id

        self.edge_processor = DiffProcessor.str_to_class(config.feature)()
        self.small_object_detection = Object_Detection(config, type='small inference')

        #create database and tables
        self.database = DataBase(config)
        self.database.use_database()
        self.database.create_tables(self.edge_id)

        self.frame_cache = Queue(config.frame_cache_maxsize)
        self.local_queue = Queue(config.local_queue_maxsize)

        #start the thread for edge server
        self.edge_server = threading.Thread(target=self.start_edge_server,)
        self.edge_server.start()
        # start the thread for diff
        self.diff_processor = threading.Thread(target=self.diff_worker,)
        self.diff_processor.start()
        # start the thread for local process
        self.local_processor = threading.Thread(target=self.local_worker,)
        self.local_processor.start()
        # start the thread pool for offload
        self.offloading_executor = futures.ThreadPoolExecutor(max_workers=config.offloading_max_worker,)


    def diff_worker(self):
        diff = 1.0
        while True:
            #get task from cache
            task = self.frame_cache.get(block=True)
            frame = task.frame_edge
            if diff == 1.0:
                #the first frame
                pre_frame_feature = self.edge_processor.get_frame_feature(frame)
                self.local_queue.put(task, block=True)
                task = self.frame_cache.get(block=True)
                frame = task.frame_edge
                diff = 0.0
            frame_feature = self.edge_processor.get_frame_feature(frame)
            # calculate and accumulate the difference
            diff += self.edge_processor.cal_frame_diff(frame_feature, pre_frame_feature)
            pre_frame_feature = frame_feature
            if diff > self.config.diff_thresh:
                diff = 0.0
                self.decision_worker(task)

    def local_worker(self):
        while True:
            #get a inference task from local queue
            current_task = self.local_queue.get(block=True)
            current_frame = current_task.frame_edge
            #get the query image and the small inference result
            small_result = self.small_object_detection.small_inference(current_frame)
            if small_result != 'object not detected':
                offloading_image, detection_boxes, detection_class, detection_score = small_result
                #add the partly small result
                if current_task.received == False:
                    current_task.add_result(detection_boxes, detection_class, detection_score)
                else:
                    # scale the small result
                    scale = current_task.raw_shape[0] / current_frame.shape[0]
                    detection_boxes = (np.array(detection_boxes) * scale).tolist()
                    current_task.add_result(detection_boxes, detection_class, detection_score)

                # whether to further inference
                if offloading_image is not None:
                    current_task.set_frame_cloud(offloading_image)
                    #offload to cloud for further inference
                    self.offloading_executor.submit(self.offload_worker, current_task)

                else:
                    end_time = time.time()
                    current_task.set_end_time(end_time)
                    #upload the result to database
                    str_result = current_task.get_result()
                    in_data = (
                        current_task.frame_index,
                        datetime.fromtimestamp(current_task.start_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
                        datetime.fromtimestamp(current_task.end_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
                        str_result,)
                    logger.debug(str(current_task.edge_id) + str(in_data))
                    self.database.insert_data(table_name='edge{}'.format(current_task.edge_id), data=in_data)
            else:
                logger.info('This frame has no objects detected')

    def offload_worker(self, current_task, destination_edge=None):
        frame_cloud = current_task.frame_cloud
        #offload to cloud or another edge
        if frame_cloud is not None:
            new_frame = frame_resize(frame_cloud, new_height=self.config.new_height)
            encoded_image = cv2_to_base64(new_frame, qp=self.config.quality)
            part_result_str = current_task.get_result()
            msg_request = message_transmission_pb2.MessageRequest(
                source_edge_id=int(self.edge_id),
                frame_index=int(current_task.frame_index),
                start_time=str(current_task.start_time),
                frame=encoded_image,
                part_result=part_result_str,
                raw_shape=str(current_task.raw_shape),
                new_shape=str(new_frame.shape),
            )
            with grpc.insecure_channel(self.config.server_ip) as channel:
                stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
                res = stub.task_processor(msg_request)
                logger.debug(str(res))
        else:
            frame_edge = current_task.frame_edge
            new_frame = frame_resize(frame_edge, new_height=self.config.new_height)
            encoded_image = cv2_to_base64(new_frame, qp=self.config.quality)
            msg_request = message_transmission_pb2.MessageRequest(
                source_edge_id=int(self.edge_id),
                frame_index=int(current_task.frame_index),
                start_time=str(current_task.start_time),
                frame=encoded_image,
                part_result="",
                raw_shape=str(current_task.raw_shape),
                new_shape=str(new_frame.shape),
            )
            destination_edge_ip = self.config.edges[destination_edge]
            with grpc.insecure_channel(destination_edge_ip) as channel:
                stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
                res = stub.task_processor(msg_request)
                logger.debug(str(res))

    def decision_worker(self, task):
        if self.config.policy == 'random policy':
            location = random.randint(0, 2)
            logger.debug(location)
            if location == 0:
                self.local_queue.put(task, block=True)
            elif location == 1:
                #directly offload to cloud
                task.set_frame_cloud(task.frame_edge)
                self.offloading_executor.submit(self.offload_worker, task)
            else:
                #offload to another edge node
                edge_num = self.config.edge_num
                other_id_list = [i for i in range(edge_num) if i+1 != self.edge_id]
                destination = random.choice(other_id_list)
                self.offloading_executor.submit(self.offload_worker, task, destination_edge=destination)

        else:
            pass

    def start_edge_server(self):
        logger.info("edge {} server is starting".format(self.edge_id))
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        message_transmission_pb2_grpc.add_MessageTransmissionServicer_to_server(
            MessageTransmissionServicer(self.local_queue, self.edge_id), server)
        server.add_insecure_port('[::]:20416')
        server.start()
        logger.success("edge {} server is listening".format(self.edge_id))
        server.wait_for_termination()