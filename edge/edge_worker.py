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

        # create database and tables
        self.database = DataBase(config)
        self.database.use_database()
        #self.database.create_tables(self.edge_id)

        self.frame_cache = Queue(config.frame_cache_maxsize)
        self.local_queue = Queue(config.local_queue_maxsize)

        # start the thread for edge server
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

        # regularly update queue information
        self.regular = BackgroundScheduler(daemon=True)
        self.regular.add_job(self.update_regular, 'interval', seconds=int(self.config.regular))
        self.regular.start()

    def diff_worker(self):
        diff = 1.0
        while True:
            # get task from cache
            task = self.frame_cache.get(block=True)
            frame = task.frame_edge
            if diff == 1.0:
                # the first frame
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
            # get a inference task from local queue
            current_task = self.local_queue.get(block=True)
            current_frame = current_task.frame_edge
            # get the query image and the small inference result
            small_result = self.small_object_detection.small_inference(current_frame)
            if small_result != 'object not detected':
                offloading_image, detection_boxes, detection_class, detection_score = small_result
                logger.debug("small_result {}".format(detection_boxes))
                logger.debug("frame index {}".format(current_task.frame_index))
                logger.debug("queue info {}".format(self.queue_info))
                # the task do not from another edge node, add the partly small result
                if current_task.other == False:
                    logger.debug('ok1')
                    current_task.add_result(detection_boxes, detection_class, detection_score)
                else:
                    logger.debug('ok2')
                    # scale the small result
                    scale = current_task.raw_shape[0] / current_frame.shape[0]
                    detection_boxes = (np.array(detection_boxes) * scale).tolist()
                    current_task.add_result(detection_boxes, detection_class, detection_score)

                # whether to further inference

                edge_process = random.choice([True, False])
                if offloading_image is not None and edge_process is False:
                    logger.debug('ok3')
                    current_task.set_frame_cloud(offloading_image)
                    # offload to cloud for further inference
                    self.offloading_executor.submit(self.offload_worker, current_task)

                else:
                    logger.debug('ok4')
                    end_time = time.time()
                    current_task.set_end_time(end_time)
                    # upload the result to database
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
        new_height = self.config.new_height
        qp = self.config.quality
        # offload to cloud or another edge
        if frame_cloud is not None:
            logger.debug("ok5")
            part_result_str = current_task.get_result()
            if current_task.directly_cloud:
                logger.debug("ok6")
                # offload to the cloud directly
                new_frame = frame_resize(frame_cloud, new_height=new_height.directly_cloud)
                encoded_image = cv2_to_base64(new_frame, qp=qp.directly_cloud)
            elif current_task.other:
                logger.debug("ok7")
                # The task from another edge node is offloaded to the cloud after local inference
                if current_task.frame_cloud.shape[0] < new_height.another_cloud:
                    logger.error("the new height can not larger than the height of current frame.")
                new_frame = frame_resize(frame_cloud, new_height=new_height.another_cloud)
                encoded_image = cv2_to_base64(new_frame, qp=qp.another_cloud)

            else:
                logger.debug("ok8")
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
            )
            with grpc.insecure_channel(self.config.server_ip) as channel:
                stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
                res = stub.task_processor(msg_request)
                logger.debug('id {}'.format(res.destination_id))
                logger.debug('cloud_reply: ' + str(res))
        else:
            logger.debug("ok9")
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
            )
            destination_edge_ip = self.config.edges[destination_edge]
            with grpc.insecure_channel(destination_edge_ip) as channel:
                stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
                res = stub.task_processor(msg_request)
                logger.debug('edge_reply: ' + str(res))

    def start_edge_server(self):
        logger.info("edge {} server is starting".format(self.edge_id))
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        message_transmission_pb2_grpc.add_MessageTransmissionServicer_to_server(
            MessageTransmissionServicer(self.local_queue, self.edge_id, self.queue_info), server)
        server.add_insecure_port('[::]:50050')
        server.start()
        logger.success("edge {} server is listening".format(self.edge_id))
        server.wait_for_termination()

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
                with grpc.insecure_channel(destination_edge_ip) as channel:
                    stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
                    res = stub.get_queue_info(info_request)
                    logger.info(res)
                    self.queue_info['{}'.format(res.destination_id)] = res.local_length

        logger.info(self.queue_info)

    def decision_worker(self, task):
        policy = self.config.policy
        self.queue_info['{}'.format(self.edge_id)] = self.local_queue.qsize()
        logger.info('the offloading policy is {}'.format(policy))
        if policy == 'Edge-Local':
            task.edge_process = True
            self.local_queue.put(task, block=True)

        elif policy == 'Edge-Shortest':
            shortest_id = min(self.queue_info, key=self.queue_info)
            task.edge_process = True
            if shortest_id == self.edge_id:
                self.local_queue.put(task, block=True)
            else:
                self.offloading_executor.submit(self.offload_worker, task, destination_edge=shortest_id)

        elif policy == 'Edge-Random':
            edge_num = self.config.edge_num
            id_list = [i for i in range(edge_num)]
            destination = random.choice(id_list)
            task.edge_process = True
            if destination == self.edge_id:
                self.local_queue.put(task, block=True)
            else:
                self.offloading_executor.submit(self.offload_worker, task, destination_edge=destination)

        elif policy == 'Edge-Cloud-Only':
            self.local_queue.put(task, block=True)

        elif policy =='Edge-Cloud-Threshold':
            queue_thresh = self.config.queue_thresh
            if self.local_queue.qsize() <= queue_thresh:
                self.local_queue.put(task, block=True)
            else:
                task.set_frame_cloud(task.frame_edge)
                task.directly_cloud = True
                self.offloading_executor.submit(self.offload_worker, task)

        elif policy == 'Edge-Cloud-Random':
            location = random.randint(0, 1)
            if location == 0:
                self.local_queue.put(task, block=True)
            else:
                # directly offload to cloud
                task.set_frame_cloud(task.frame_edge)
                task.directly_cloud = True
                self.offloading_executor.submit(self.offload_worker, task)

        elif policy == 'Shortest-Queue-Only':
            shortest_id = min(self.queue_info, key=self.queue_info)
            if shortest_id == self.edge_id:
                self.local_queue.put(task, block=True)
            else:
                self.offloading_executor.submit(self.offload_worker, task, destination_edge=shortest_id)

        elif policy == 'Shortest-Queue-Threshold':
            queue_thresh = self.config.queue_thresh
            shortest_info = min(zip(self.queue_info.values(), self.queue_info.keys()))
            shortest_length = shortest_info[0]
            shortest_id = shortest_info[1]
            if shortest_length > queue_thresh:
                task.set_frame_cloud(task.frame_edge)
                task.directly_cloud = True
                self.offloading_executor.submit(self.offload_worker, task)
            elif shortest_id == self.edge_id:
                self.local_queue.put(task, block=True)
            else:
                self.offloading_executor.submit(self.offload_worker, task, destination_edge=shortest_id)

        elif policy == 'Random-Only':
            edge_num = self.config.edge_num
            id_list = [i for i in range(edge_num)]
            destination = random.choice(id_list)
            if destination == self.edge_id:
                self.local_queue.put(task, block=True)
            else:
                self.offloading_executor.submit(self.offload_worker, task, destination_edge=destination)

        elif policy == 'Random-Threshold':
            queue_thresh = self.config.queue_thresh
            edge_num = self.config.edge_num
            id_list = [i for i in range(edge_num)]
            destination = random.choice(id_list)
            des_len = self.queue_info['{}'.format(destination)]

            if des_len > queue_thresh:
                task.set_frame_cloud(task.frame_edge)
                task.directly_cloud = True
                self.offloading_executor.submit(self.offload_worker, task)
            elif destination == self.edge_id:
                self.local_queue.put(task, block=True)
            else:
                self.offloading_executor.submit(self.offload_worker, task, destination_edge=destination)

        else:
            logger.error('the policy does not exist.')

