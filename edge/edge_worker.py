import threading
import time
from concurrent import futures
from datetime import datetime

import random

import cv2
import grpc
import numpy as np
from queue import Queue
from loguru import logger

from database.database import DataBase
from difference.diff import DiffProcessor
from grpc_server import message_transmission_pb2_grpc, message_transmission_pb2
from grpc_server.rpc_server import MessageTransmissionServicer

from tools.convert_tool import cv2_to_base64
from tools.preprocess import frame_resize
from inferencer.object_detection import Object_Detection
from apscheduler.schedulers.background import BackgroundScheduler



class EdgeWorker:
    def __init__(self, config):

        self.config = config
        self.edge_id = config.edge_id
        # queue info (id: length)
        self.edge_num = self.config.edge_num
        self.queue_info = {'{}'.format(i+1): 0 for i in range(self.edge_num)}

        self.edge_processor = DiffProcessor.str_to_class(config.feature)()
        self.small_object_detection = Object_Detection(config, type='small inference')
        #test_frame = cv2.imread('test.jpg')
        #self.small_object_detection.small_inference(test_frame)

        # create database and tables
        self.database = DataBase(config)
        self.database.use_database()
        self.database.clear_table(self.edge_id)

        self.frame_cache = Queue(config.frame_cache_maxsize)
        self.local_queue = Queue(config.local_queue_maxsize)
        self.upload_data = Queue(config.frame_cache_maxsize)

        # start the thread for edge server
        if self.edge_num > 1:
            self.edge_server = threading.Thread(target=self.start_edge_server,)
            self.edge_server.start()
        # start the thread for diff
        self.diff_processor = threading.Thread(target=self.diff_worker,)
        self.diff_processor.start()
        # start the thread for local process
        self.local_processor = threading.Thread(target=self.local_worker,)
        self.local_processor.start()
        #start the thread for uploading data
        self.upload_processor = threading.Thread(target=self.uploading_worker,)
        self.upload_processor.start()
        # start the thread pool for offload
        self.offloading_executor = futures.ThreadPoolExecutor(max_workers=config.offloading_max_worker,)

        # regularly update queue information
        self.regular = BackgroundScheduler(daemon=True)
        self.regular.add_job(self.update_regular, 'interval', seconds=int(self.config.regular))
        self.regular.start()

    def diff_worker(self):
        logger.info('the offloading policy is {}'.format(self.config.policy))
        diff_first = True
        while True:
            # get task from cache
            task = self.frame_cache.get(block=True)
            frame = task.frame_edge
            if self.config.diff_lable is True:
                if diff_first:
                    # the first frame
                    diff_first = False
                    pre_frame_feature = self.edge_processor.get_frame_feature(frame)
                    self.decision_worker(task)
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
            else:
                self.decision_worker(task)

    def local_worker(self):
        while True:
            # get a inference task from local queue
            current_task = self.local_queue.get(block=True)
            current_time = time.time()
            while current_time - current_task.start_time > self.config.timeout_drop:
                current_task.time_out = True
                in_data = (
                    current_task.frame_index,
                    datetime.fromtimestamp(current_task.start_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
                    datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
                    "",
                    "Timeout",)

                self.upload_data.put((current_task.edge_id, in_data), block=True)
                current_task = self.local_queue.get(block=True)
                current_time = time.time()

            self.queue_info['{}'.format(self.edge_id)] = self.local_queue.qsize()
            current_frame = current_task.frame_edge
            # get the query image and the small inference result
            small_result = self.small_object_detection.small_inference(current_frame)
            if small_result != 'object not detected':
                offloading_image, detection_boxes, detection_class, detection_score = small_result
                # the task do not from another edge node, add the partly small result
                if current_task.other == False:
                    current_task.add_result(detection_boxes, detection_class, detection_score)
                else:
                    # scale the small result
                    scale = current_task.raw_shape[0] / current_frame.shape[0]
                    if detection_boxes is not None:
                        detection_boxes = (np.array(detection_boxes) * scale).tolist()
                    current_task.add_result(detection_boxes, detection_class, detection_score)

                # whether to further inference
                if offloading_image is not None and current_task.edge_process is False:
                    current_task.set_frame_cloud(offloading_image)
                    # offload to cloud for further inference
                    self.offloading_executor.submit(self.offload_worker, current_task)

                else:
                    end_time = time.time()
                    current_task.set_end_time(end_time)
                    # upload the result to database
                    str_result = current_task.get_result()
                    in_data = (
                        current_task.frame_index,
                        datetime.fromtimestamp(current_task.start_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
                        datetime.fromtimestamp(current_task.end_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
                        str_result,
                        "",
                    )

                    self.upload_data.put((current_task.edge_id, in_data), block=True)
            else:
                end_time = time.time()
                current_task.set_end_time(end_time)
                # upload the result to database
                in_data = (
                    current_task.frame_index,
                    datetime.fromtimestamp(current_task.start_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
                    datetime.fromtimestamp(current_task.end_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
                    "",
                    "",
                    )
                self.upload_data.put((current_task.edge_id, in_data), block=True)
                logger.info('This frame has no objects detected')

    def offload_worker(self, current_task, destination_edge=None):
        frame_cloud = current_task.frame_cloud
        new_height = self.config.new_height
        qp = self.config.quality
        # offload to cloud or another edge
        if frame_cloud is not None:
            part_result_str = current_task.get_result()
            if current_task.directly_cloud:
                # offload to the cloud directly
                new_frame = frame_resize(frame_cloud, new_height=new_height.directly_cloud)
                encoded_image = cv2_to_base64(new_frame, qp=qp.directly_cloud)
            elif current_task.other:
                # The task from another edge node is offloaded to the cloud after local inference
                if current_task.frame_cloud.shape[0] < new_height.another_cloud:
                    logger.error("the new height can not larger than the height of current frame.")
                new_frame = frame_resize(frame_cloud, new_height=new_height.another_cloud)
                encoded_image = cv2_to_base64(new_frame, qp=qp.another_cloud)

            else:
                # offload to the cloud after local inference
                new_frame = frame_resize(frame_cloud, new_height=new_height.local_cloud)
                encoded_image = cv2_to_base64(new_frame, qp=qp.local_cloud)

            msg_request = message_transmission_pb2.MessageRequest(
                source_edge_id=int(self.edge_id),
                frame_index=int(current_task.frame_index),
                start_time=str(current_task.start_time),
                frame=encoded_image,
                part_result=part_result_str,
                raw_shape=str(current_task.raw_shape),
                new_shape=str(new_frame.shape),
                note="",
            )
            try:
                channel = grpc.insecure_channel(self.config.server_ip)
                stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
                res = stub.task_processor(msg_request)
            except Exception as e:
                logger.exception("the cloud can not reply, {}".format(e))
                #put the task into local queue
                if current_task.directly_cloud:
                    current_task.edge_process = True
                    self.local_queue.put(current_task, block=True)
                #upload the result
                else:
                    end_time = time.time()
                    current_task.set_end_time(end_time)
                    in_data = (
                        current_task.frame_index,
                        datetime.fromtimestamp(current_task.start_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
                        datetime.fromtimestamp(current_task.end_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
                        part_result_str,
                        "",
                    )

                    self.upload_data.put((current_task.edge_id, in_data), block=True)

            else:
                self.queue_info['{}'.format(self.edge_id)] = self.local_queue.qsize()

        else:
            frame_edge = current_task.frame_edge
            new_frame = frame_resize(frame_edge, new_height=new_height.another)
            encoded_image = cv2_to_base64(new_frame, qp=qp.another)
            msg_request = message_transmission_pb2.MessageRequest(
                source_edge_id=int(self.edge_id),
                frame_index=int(current_task.frame_index),
                start_time=str(current_task.start_time),
                frame=encoded_image,
                part_result="",
                raw_shape=str(current_task.raw_shape),
                new_shape=str(new_frame.shape),
                note="edge process",
            )
            destination_edge_ip = self.config.edges[destination_edge]
            try:
                channel = grpc.insecure_channel(destination_edge_ip)
                stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
                res = stub.task_processor(msg_request)
            except Exception as e:
                logger.exception("the edge ip {} can not reply".format(destination_edge_ip))
                self.local_queue.put(current_task, block=True)
            else:
                logger.info(str(res))
                self.queue_info['{}'.format(self.edge_id)] = self.local_queue.qsize()
                self.queue_info['{}'.format(res.destination_id)] = res.local_length

    def start_edge_server(self):
        logger.info("edge {} server is starting".format(self.edge_id))
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        message_transmission_pb2_grpc.add_MessageTransmissionServicer_to_server(
            MessageTransmissionServicer(self.local_queue, self.edge_id, self.queue_info), server)
        server.add_insecure_port('[::]:50050')
        server.start()
        logger.success("edge {} server is listening".format(self.edge_id))
        server.wait_for_termination()

    def uploading_worker(self):
        while True:
            source_id, insert_data = self.upload_data.get(block=True)
            self.database.insert_data(table_name='edge{}'.format(source_id), data=insert_data)
            logger.debug("insert successfully " + str(source_id) + " " + str(insert_data))

    def update_regular(self):
        for i in range(self.edge_num):
            if i+1 == self.edge_id:
                self.queue_info['{}'.format(self.edge_id)] = self.local_queue.qsize()
            else:
                info_request = message_transmission_pb2.InfoRequest(
                    source_edge_id=self.edge_id,
                    local_length=self.local_queue.qsize(),
                )
                destination_edge_ip = self.config.edges[i]
                channel = grpc.insecure_channel(destination_edge_ip)
                stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
                try:
                    res = stub.get_queue_info(info_request)
                except Exception as e:
                    logger.exception("the edge ip {} can not reply".format(destination_edge_ip))
                    self.queue_info['{}'.format(self.edge_id)] = self.local_queue.qsize()
                else:
                    logger.info(res)
                    self.queue_info['{}'.format(self.edge_id)] = self.local_queue.qsize()
                    self.queue_info['{}'.format(res.destination_id)] = res.local_length
        logger.info(self.queue_info)

    def decision_worker(self, task):
        policy = self.config.policy
        self.queue_info['{}'.format(self.edge_id)] = self.local_queue.qsize()
        if policy == 'Edge-Local':
            task.edge_process = True
            self.local_queue.put(task, block=True)

        elif policy == 'Edge-Shortest':
            shortest_id = min(self.queue_info, key=self.queue_info.get)
            task.edge_process = True
            if int(shortest_id) == self.edge_id or \
                    self.queue_info['{}'.format(shortest_id)] == self.queue_info['{}'.format(self.edge_id)]:
                self.local_queue.put(task, block=True)
            else:
                self.offloading_executor.submit(self.offload_worker, task, destination_edge=(int(shortest_id)-1))

        elif policy == 'Edge-Random':
            edge_num = self.config.edge_num
            id_list = [i for i in range(edge_num)]
            destination = random.choice(id_list)
            task.edge_process = True
            if destination == self.edge_id:
                self.local_queue.put(task, block=True)
            else:
                self.offloading_executor.submit(self.offload_worker, task, destination_edge=destination)

        elif policy == 'Edge-Cloud-Assisted':
            self.local_queue.put(task, block=True)

        elif policy =='Edge-Cloud-Threshold':
            queue_thresh = self.config.queue_thresh
            if self.local_queue.qsize() <= queue_thresh:
                task.edge_process = True
                self.local_queue.put(task, block=True)
            else:
                task.set_frame_cloud(task.frame_edge)
                task.directly_cloud = True
                self.offloading_executor.submit(self.offload_worker, task)


        elif policy == 'Shortest-Queue-Assisted':
            shortest_id = min(self.queue_info, key=self.queue_info.get)
            if int(shortest_id) == self.edge_id:
                self.local_queue.put(task, block=True)
            else:
                self.offloading_executor.submit(self.offload_worker, task, destination_edge=int(shortest_id)-1)

        elif policy == 'Shortest-Queue-Threshold':
            queue_thresh = self.config.queue_thresh
            shortest_info = min(zip(self.queue_info.values(), self.queue_info.keys()))
            shortest_length = shortest_info[0]
            shortest_id = shortest_info[1]
            if shortest_length > queue_thresh:
                task.set_frame_cloud(task.frame_edge)
                task.directly_cloud = True
                self.offloading_executor.submit(self.offload_worker, task)
            elif int(shortest_id) == self.edge_id:
                task.edge_process = True
                self.local_queue.put(task, block=True)
            else:
                task.edge_process = True
                self.offloading_executor.submit(self.offload_worker, task, destination_edge=int(shortest_id)-1)

        else:
            logger.error('the policy does not exist.')

