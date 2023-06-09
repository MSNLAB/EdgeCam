import os
import sys
import threading
import time
from concurrent import futures

import random

import cv2
import grpc
import numpy as np
from queue import Queue
from loguru import logger

from database.database import DataBase
from difference.diff import DiffProcessor
from edge.info import TASK_STATE
from edge.resample import history_sample, annotion_process
from edge.task import Task
from edge.transmit import get_cloud_target, is_network_connected
from grpc_server import message_transmission_pb2_grpc, message_transmission_pb2
from grpc_server.rpc_server import MessageTransmissionServicer

from tools.convert_tool import cv2_to_base64
from tools.file_op import clear_folder, creat_folder, sample_files
from tools.preprocess import frame_resize
from model_management.object_detection import Object_Detection
from apscheduler.schedulers.background import BackgroundScheduler

from tools.video_processor import VideoProcessor




class EdgeWorker:
    def __init__(self, config):

        self.config = config
        self.edge_id = config.edge_id

        self.edge_processor = DiffProcessor.str_to_class(config.feature)()
        self.small_object_detection = Object_Detection(config, type='small inference')

        # create database and tables
        self.database = DataBase(self.config.database)
        self.database.use_database()
        self.database.clear_table(self.edge_id)
        self.data_lock = threading.Lock()

        self.queue_info = {'{}'.format(i + 1): 0 for i in range(self.config.edge_num)}
        self.frame_cache = Queue(config.frame_cache_maxsize)
        self.local_queue = Queue(config.local_queue_maxsize)

        self.pred_res = []
        self.avg_scores = []
        self.select_index = []
        self.annotations = []

        # start the thread for diff
        self.diff = 0
        self.key_task = None
        self.diff_processor = threading.Thread(target=self.diff_worker,daemon=True)
        self.diff_processor.start()

        # start the thread for edge server
        #self.edge_server = threading.Thread(target=self.start_edge_server, daemon=True)
        #self.edge_server.start()

        # start the thread for local process
        self.local_processor = threading.Thread(target=self.local_worker,daemon=True)
        self.local_processor.start()

        # start the thread for retrain process
        self.collect_flag = self.config.retrain.flag
        self.collect_time = None
        self.collect_time_flag = True
        self.retrain_flag = False
        self.cache_count = 0

        self.use_history = True
        self.retrain_no = 0

        self.retrain_processor = threading.Thread(target=self.retrain_worker,daemon=True)
        self.retrain_processor.start()

        # start the thread pool for offload
        self.offloading_executor = futures.ThreadPoolExecutor(max_workers=config.offloading_max_worker,)




    def diff_worker(self):
        logger.info('the offloading policy is {}'.format(self.config.policy))
        if self.config.diff_flag:
            task = self.frame_cache.get(block=True)
            frame = task.frame_edge
            self.pre_frame_feature = self.edge_processor.get_frame_feature(frame)
            self.key_task = task
            # Create an entry for the task in the database table
            data = (
                task.frame_index,
                task.start_time,
                None,
                "",
                "",)
            with self.data_lock:
                self.database.insert_data(self.edge_id, data)
            task.edge_process = True
            self.local_queue.put(task, block=True)

            while True:
                # get task from cache
                task = self.frame_cache.get(block=True)
                # Create an entry for the task in the database table
                data = (
                    task.frame_index,
                    task.start_time,
                    None,
                    "",
                    "",)
                with self.data_lock:
                    self.database.insert_data(self.edge_id, data)

                frame = task.frame_edge
                self.frame_feature = self.edge_processor.get_frame_feature(frame)
                # calculate and accumulate the difference
                self.diff += self.edge_processor.cal_frame_diff(self.frame_feature, self.pre_frame_feature)
                self.pre_frame_feature = self.frame_feature
                # Process the video frame greater than a certain threshold
                if self.diff >= self.config.diff_thresh:
                    self.diff = 0.0
                    self.key_task = task
                    self.decision_worker(task)
                else:
                    task.end_time = time.time()
                    task.ref = self.key_task.frame_index
                    task.state = TASK_STATE.FINISHED
                    self.update_table(task)

    def update_table(self, task):
        if task.state == TASK_STATE.FINISHED:
            state = "Finished"
        elif task.state == TASK_STATE.TIMEOUT:
            state = "Timeout"
        else:
            state = ""
        if task.ref is not None:
            result = {'ref': task.ref}
        else:
            detection_boxes, detection_class, detection_score = task.get_result()
            result = {
                'labels': detection_class,
                'boxes': detection_boxes,
                'scores': detection_score
            }
        # upload the result to database
        data = (
            task.end_time,
            str(result),
            state,
            task.frame_index)
        with self.data_lock:
            self.database.update_data(task.edge_id, data)


    def local_worker(self):
        while True:
            # get a inference task from local queue
            task = self.local_queue.get(block=True)
            if time.time() - task.start_time >= self.config.wait_thresh:
                end_time = time.time()
                task.end_time = end_time
                task.state = TASK_STATE.TIMEOUT
                self.update_table(task)
                continue
            self.queue_info['{}'.format(self.edge_id)] = self.local_queue.qsize()
            current_frame = task.frame_edge
            # get the query image and the small inference result
            offloading_image, detection_boxes, detection_class, detection_score \
                = self.small_object_detection.small_inference(current_frame)

            # collect data for retrain
            if self.collect_flag:
                if self.collect_time_flag:
                    self.collect_time = time.time()
                    self.collect_time_flag = False
                duration = time.time() - self.collect_time
                if duration > self.config.retrain.window:
                    self.collect_data(task, current_frame ,detection_boxes, detection_class, detection_score)

            if detection_boxes is not None:
                task.add_result(detection_boxes, detection_class, detection_score)
                # whether to further inference
                if offloading_image is not None and task.edge_process is False:
                    # offload to cloud for further inference
                    task.frame_cloud = offloading_image
                    self.offloading_executor.submit(self.offload_worker, task)
                # local infer, upload result
                else:
                    end_time = time.time()
                    task.end_time = end_time
                    task.state = TASK_STATE.FINISHED
                    self.update_table(task)
            else:
                if offloading_image is not None and task.edge_process is False:
                    # offload to cloud for further inference
                    task.frame_cloud = offloading_image
                    self.offloading_executor.submit(self.offload_worker, task)
                end_time = time.time()
                task.end_time = end_time
                task.state = TASK_STATE.FINISHED
                # upload the result to database
                self.update_table(task)
                logger.info('target not detected')

    def offload_worker(self, task, destination_edge_id=None):
        new_height = self.config.new_height
        qp = self.config.quality
        # offload to cloud
        if destination_edge_id is None:
            frame_cloud = task.frame_cloud
            assert frame_cloud is not None
            detection_boxes, detection_class, detection_score = task.get_result()
            if len(detection_boxes) != 0:
                part_result = {'boxes': detection_boxes, 'labels': detection_class, 'scores': detection_score}
            else:
                part_result = ""
            # offload to the cloud directly
            if task.directly_cloud:
                new_frame = frame_resize(frame_cloud, new_height=new_height.directly_cloud)
                encoded_image = cv2_to_base64(new_frame, qp=qp.directly_cloud)
            # The task from another edge node is offloaded to the cloud after local inference
            elif task.other:
                if task.frame_cloud.shape[0] < new_height.another_cloud:
                    logger.error("the new height can not larger than the height of current frame.")
                new_frame = frame_resize(frame_cloud, new_height=new_height.another_cloud)
                encoded_image = cv2_to_base64(new_frame, qp=qp.another_cloud)
            # offload to the cloud after local inference
            else:
                new_frame = frame_resize(frame_cloud, new_height=new_height.local_cloud)
                encoded_image = cv2_to_base64(new_frame, qp=qp.local_cloud)

            msg_request = message_transmission_pb2.MessageRequest(
                source_edge_id=int(self.edge_id),
                frame_index=int(task.frame_index),
                start_time=str(task.start_time),
                frame=encoded_image,
                part_result=str(part_result),
                raw_shape=str(task.raw_shape),
                new_shape=str(new_frame.shape),
                note="",
            )
            try:
                channel = grpc.insecure_channel(self.config.server_ip)
                stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
                res = stub.task_processor(msg_request)
            except Exception as e:
                logger.exception("the cloud can not reply, {}".format(e))
                # put the task into local queue
                if task.directly_cloud:
                    task.edge_process = True
                    self.local_queue.put(task, block=True)
                # upload the result
                else:
                    end_time = time.time()
                    task.end_time = end_time
                    self.update_table(task)
            else:
                self.queue_info['{}'.format(self.edge_id)] = self.local_queue.qsize()

        # to another edge
        else:
            frame_edge = task.frame_edge
            new_frame = frame_resize(frame_edge, new_height=new_height.another)
            encoded_image = cv2_to_base64(new_frame, qp=qp.another)
            if task.edge_process is True:
                note = "edge process"
            msg_request = message_transmission_pb2.MessageRequest(
                source_edge_id=int(self.edge_id),
                frame_index=int(task.frame_index),
                start_time=str(task.start_time),
                frame=encoded_image,
                part_result="",
                raw_shape=str(task.raw_shape),
                new_shape=str(new_frame.shape),
                note=note,
            )
            destinations = self.config.destinations
            destination_edge_ip = destinations['ip'][destinations['id'] == destination_edge_id]
            try:
                channel = grpc.insecure_channel(destination_edge_ip)
                stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
                res = stub.task_processor(msg_request)
            except Exception as e:
                logger.exception("the edge id{}, ip {} can not reply, {}".format(destination_edge_id,destination_edge_ip,e))
                self.local_queue.put(task, block=True)
            else:
                logger.debug("forward to other edge")
                self.queue_info['{}'.format(self.edge_id)] = self.local_queue.qsize()
                self.queue_info['{}'.format(res.destination_edge_id)] = res.local_length

    # collect data for retrain
    def collect_data(self, task, frame ,detection_boxes, detection_class, detection_score):
        if detection_score is not None:
            creat_folder(self.config.retrain.cache_path)
            cv2.imwrite(os.path.join(self.config.retrain.cache_path,'frames', str(task.frame_index) + '.jpg'), frame)
            self.avg_scores.append({task.frame_index: np.mean(detection_score)})
            self.cache_count += 1
            if self.cache_count >= self.config.retrain.collect_num:
                self.retrain_no += 1
                smallest_elements = sorted(self.avg_scores, key=lambda d: list(d.values())[0])[:self.config.retrain.select_num]
                self.select_index = [list(d.keys())[0] for d in smallest_elements]
                logger.debug("the select index {}".format(self.select_index))
                self.pred_res = []
                self.collect_flag = False
                self.cache_count = 0
                self.retrain_flag = True


    # retrain
    def retrain_worker(self):
        self.annotations = []
        while True:
            if self.retrain_flag:
                logger.debug("retrain")
                for index in self.select_index:
                    path = os.path.join(self.config.retrain.cache_path, 'frames', '{}.jpg'.format(index))
                    logger.debug(path)
                    frame = cv2.imread(path)
                    target_res = get_cloud_target(self.config.server_ip, frame)
                    for score, label, box in zip(target_res['scores'], target_res['labels'], target_res['boxes']):
                        self.annotations.append((index, label, box[0], box[1], box[2], box[3], score))
                if len(self.annotations):
                    np.savetxt(os.path.join(self.config.retrain.cache_path,'annotation.txt'), self.annotations,
                               fmt=['%d', '%d', '%f', '%f', '%f', '%f', '%f'], delimiter=',')

                self.small_object_detection.model_evaluation(
                    self.config.retrain.cache_path, self.select_index)
                self.small_object_detection.retrain(
                    self.config.retrain.cache_path, self.select_index[:int(self.config.retrain.select_num*0.8)])
                self.retrain_flag = False
                if self.use_history:
                    self.select_index,self.avg_scores = history_sample(self.select_index,self.avg_scores)
                    self.annotations = annotion_process(self.annotations, self.select_index)
                    sample_files(os.path.join(self.config.retrain.cache_path, 'frames') ,self.select_index)
                    self.cache_count = len(self.select_index)
                else:
                    clear_folder(self.config.retrain.cache_path)
                    self.select_index = []
                    self.avg_scores = []
                    self.annotations = []
                self.collect_time_flag = True
                self.collect_flag = True
            time.sleep(0.2)



    def start_edge_server(self):
        logger.info("edge {} server is starting".format(self.edge_id))
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        message_transmission_pb2_grpc.add_MessageTransmissionServicer_to_server(
            MessageTransmissionServicer(self.local_queue, self.edge_id, self.small_object_detection ), server)
        server.add_insecure_port('[::]:50050')
        server.start()
        logger.success("edge {} server is listening".format(self.edge_id))
        server.wait_for_termination()


    def decision_worker(self, task):
        policy = self.config.policy
        if policy == 'Edge-Local':
            task.edge_process = True
            self.local_queue.put(task, block=True)

        elif policy == 'Edge-Cloud-Assisted':
            self.local_queue.put(task, block=True)

        elif policy =='Edge-Cloud-Threshold':
            queue_thresh = self.config.queue_thresh
            if self.local_queue.qsize() <= queue_thresh:
                task.edge_process = True
                self.local_queue.put(task, block=True)
            else:
                task.frame_cloud = task.frame_edge
                task.directly_cloud = True
                self.offloading_executor.submit(self.offload_worker, task)

        elif policy == 'Edge-Shortest':
            shortest_id = min(self.queue_info, key=self.queue_info.get)
            task.edge_process = True
            if int(shortest_id) == self.edge_id or \
                    self.queue_info['{}'.format(shortest_id)] == self.queue_info['{}'.format(self.edge_id)]:
                self.local_queue.put(task, block=True)
            else:
                destinations = self.config.destinations
                destination_edge_ip = destinations['ip'][destinations['id'] == shortest_id]
                if is_network_connected(destination_edge_ip):
                    self.offloading_executor.submit(self.offload_worker, task, int(shortest_id))
                else:
                    logger.info("could not connect to {}".format(destination_edge_ip))
                    self.local_queue.put(task, block=True)

        elif policy == 'Shortest-Queue-Threshold':
            queue_thresh = self.config.queue_thresh
            shortest_info = min(zip(self.queue_info.values(), self.queue_info.keys()))
            shortest_length = shortest_info[0]
            shortest_id = shortest_info[1]
            if shortest_length > queue_thresh:
                task.frame_cloud = task.frame_edge
                task.directly_cloud = True
                self.offloading_executor.submit(self.offload_worker, task)
            elif int(shortest_id) == self.edge_id:
                task.edge_process = True
                self.local_queue.put(task, block=True)
            else:
                task.edge_process = True
                self.offloading_executor.submit(self.offload_worker, task, int(shortest_id))

        else:
            logger.error('the policy does not exist.')

