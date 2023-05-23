import os
import sys
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
from edge.info import TASK_STATE
from edge.task import Task
from grpc_server import message_transmission_pb2_grpc, message_transmission_pb2
from grpc_server.rpc_server import MessageTransmissionServicer

from tools.convert_tool import cv2_to_base64
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

        self.frame_cache = Queue(config.frame_cache_maxsize)
        self.local_queue = Queue(config.local_queue_maxsize)

        self.pred_res = []
        self.avg_scores = []
        self.select_index = []
        self.annotations = []
        # read video
        self.video_reader = threading.Thread(target=self.video_read,)
        self.video_reader.start()
        # start the thread for diff
        self.diff = 0
        self.key_task = None
        self.diff_processor = threading.Thread(target=self.diff_worker,)
        self.diff_processor.start()
        # start the thread for local process
        self.local_processor = threading.Thread(target=self.local_worker,)
        self.local_processor.start()

        # start the thread for retrain process
        self.retrain_flag = False
        self.collect_flag = True
        self.cache_count = 0
        self.retrain_processor = threading.Thread(target=self.retrain_worker,)
        self.retrain_processor.start()

        # start the thread pool for offload
        self.offloading_executor = futures.ThreadPoolExecutor(max_workers=config.offloading_max_worker,)




    def video_read(self):
        with VideoProcessor(self.config.source) as video:
            video_fps = video.fps
            logger.info("the video fps is {}".format(video_fps))
            index = 0
            if self.config.interval == 0:
                logger.error("the interval error")
                sys.exit(1)
            logger.info("Take the frame interval is {}".format(self.config.interval))
            while True:
                frame = next(video)
                if frame is None:
                    logger.info("the video finished")
                    break
                index += 1
                if index % self.config.interval == 0:
                    start_time = time.time()
                    task = Task(self.edge_id, index, frame, start_time, frame.shape)
                    self.frame_cache.put(task, block=True)
                    time.sleep((self.config.interval * 1.0) / video_fps)


    def diff_worker(self):
        logger.info('the offloading policy is {}'.format(self.config.policy))
        task = self.frame_cache.get(block=True)
        # Create an entry for the task in the database table
        data = (
            task.frame_index,
            datetime.fromtimestamp(task.start_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
            None,
            "",
            "",)
        self.database.insert_data(self.edge_id, data)

        if self.config.diff_flag:
            frame = task.frame_edge
            self.pre_frame_feature = self.edge_processor.get_frame_feature(frame)
            self.key_task = task
            # local inference
            self.local_queue.put(task, block=True)

            while True:
                # get task from cache
                task = self.frame_cache.get(block=True)

                # Create an entry for the task in the database table
                data = (
                    task.frame_index,
                    datetime.fromtimestamp(task.start_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
                    None,
                    "",
                    "",)
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
                    row = self.database.select_one_result(self.edge_id, task.frame_index)
                    logger.debug(row)
                    if row[4] == 'FINISHED':
                        self.key_task.state = TASK_STATE.FINISHED
                        result_dict = eval(row[3])
                        self.key_task.add_result(result_dict['boxes'],result_dict['labels'],result_dict['scores'])
                    if self.key_task.state == TASK_STATE.FINISHED:
                        detection_boxes, detection_class, detection_score = task.get_result()
                        task.add_result(detection_boxes, detection_class, detection_score)
                        task.state = TASK_STATE.FINISHED
                        self.update_table(task)
                    else:
                        self.key_task.ref_list.append(task)

        else:
            self.decision_worker(task)

    def update_table(self, task):
        detection_boxes, detection_class, detection_score = task.get_result()
        result = {
            'lables': detection_class,
            'boxes': detection_boxes,
            'scores': detection_score
        }
        # upload the result to database
        data = (
            datetime.fromtimestamp(task.end_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
            str(result),
            "",
            task.frame_index)
        self.database.update_data(self.edge_id, data)

    def ref_update(self, task):
        for ref_task in task.ref_list:
            detection_boxes, detection_class, detection_score = task.get_result()
            ref_task.add_result(detection_boxes, detection_class, detection_score)
            ref_task.state = TASK_STATE.FINISHED
            self.update_table(task)

    def local_worker(self):
        while True:
            # get a inference task from local queue
            current_task = self.local_queue.get(block=True)
            current_frame = current_task.frame_edge
            # get the query image and the small inference result
            offloading_image, detection_boxes, detection_class, detection_score \
                = self.small_object_detection.small_inference(current_frame)
            logger.debug("img{}, box{}, cls{}, score{}".format
                         (offloading_image,detection_boxes,detection_class,detection_score))
            # collect data for retrain
            if self.collect_flag:
                self.collect_data(current_task, current_frame ,detection_boxes, detection_class, detection_score)

            if offloading_image != None and detection_boxes != None:
                current_task.add_result(detection_boxes, detection_class, detection_score)
                # whether to further inference
                if offloading_image is not None and current_task.edge_process is False:
                    # offload to cloud for further inference
                    current_task.frame_cloud = offloading_image
                    self.offloading_executor.submit(self.offload_worker, current_task)
                # local infer, upload result
                else:
                    end_time = time.time()
                    current_task.end_time = end_time
                    current_task.state = TASK_STATE.FINISHED
                    self.update_table(current_task)
                    self.ref_update(current_task)
            else:
                end_time = time.time()
                current_task.set_end_time(end_time)
                # upload the result to database
                self.update_table(current_task)
                logger.info('target not detected')

    def offload_worker(self, current_task, destination_edge_id=None):
        new_height = self.config.new_height
        qp = self.config.quality
        # offload to cloud
        if destination_edge_id is None:
            frame_cloud = current_task.frame_cloud
            assert frame_cloud != None
            part_result_str = current_task.get_result()

            # offload to the cloud directly
            if current_task.directly_cloud:
                new_frame = frame_resize(frame_cloud, new_height=new_height.directly_cloud)
                encoded_image = cv2_to_base64(new_frame, qp=qp.directly_cloud)
            # The task from another edge node is offloaded to the cloud after local inference
            elif current_task.other:
                if current_task.frame_cloud.shape[0] < new_height.another_cloud:
                    logger.error("the new height can not larger than the height of current frame.")
                new_frame = frame_resize(frame_cloud, new_height=new_height.another_cloud)
                encoded_image = cv2_to_base64(new_frame, qp=qp.another_cloud)
            # offload to the cloud after local inference
            else:
                new_frame = frame_resize(frame_cloud, new_height=new_height.local_cloud)
                encoded_image = cv2_to_base64(new_frame, qp=qp.local_cloud)

            ref_dict = {'index':[], 'start_time':[], 'end_time':[]}
            for task in current_task.ref_list:
                ref_dict['index'].append(task.frame_index)
                ref_dict['start_time'].append(task.start_time)
                ref_dict['end_time'].append(task.end_time)
            msg_request = message_transmission_pb2.MessageRequest(
                source_edge_id=int(self.edge_id),
                frame_index=int(current_task.frame_index),
                start_time=str(current_task.start_time),
                frame=encoded_image,
                part_result=part_result_str,
                raw_shape=str(current_task.raw_shape),
                new_shape=str(new_frame.shape),
                ref_list = str(ref_dict),
                note="",
            )
            try:
                channel = grpc.insecure_channel(self.config.server_ip)
                stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
                res = stub.task_processor(msg_request)
            except Exception as e:
                logger.exception("the cloud can not reply, {}".format(e))
                # put the task into local queue
                if current_task.directly_cloud:
                    current_task.edge_process = True
                    self.local_queue.put(current_task, block=True)
                # upload the result
                else:
                    end_time = time.time()
                    current_task.end_time = end_time
                    self.update_table(current_task)
            else:
                logger.info(str(res))
        # to another edge
        else:
            frame_edge = current_task.frame_edge
            new_frame = frame_resize(frame_edge, new_height=new_height.another)
            encoded_image = cv2_to_base64(new_frame, qp=qp.another)

            ref_dict = {'index': [], 'start_time': [], 'end_time': []}
            for task in current_task.ref_list:
                ref_dict['index'].append(task.frame_index)
                ref_dict['start_time'].append(task.start_time)
                ref_dict['end_time'].append(task.end_time)

            msg_request = message_transmission_pb2.MessageRequest(
                source_edge_id=int(self.edge_id),
                frame_index=int(current_task.frame_index),
                start_time=str(current_task.start_time),
                frame=encoded_image,
                part_result="",
                raw_shape=str(current_task.raw_shape),
                new_shape=str(new_frame.shape),
                ref_list=str(ref_dict),
                note="edge process",
            )
            destinations = self.config.destinations
            destination_edge_ip = destinations[destinations['id'] == destination_edge_id]
            try:
                channel = grpc.insecure_channel(destination_edge_ip)
                stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
                res = stub.task_processor(msg_request)
            except Exception as e:
                logger.exception("the edge id{}, ip {} can not reply, {}".format(destination_edge_id,destination_edge_ip,e))
                self.local_queue.put(current_task, block=True)
            else:
                logger.info(str(res))

    #
    def collect_data(self, task, frame ,detection_boxes, detection_class, detection_score):
        self.select_index = []
        cv2.imwrite(os.path.join(self.config.retrain.cache_path, str(task.frame_index) + '.jpg'), frame)
        self.avg_scores.append({task.frame_index:np.mean(detection_score)})
        self.cache_count += 1
        for score, label, box in zip(detection_score, detection_class, detection_boxes):
            logger.debug("score {}, lable {}, box {}".format(score, label, box))
            self.pred_res.append((task.frame_index, label, box[0], box[1], box[2], box[3], score))
            if self.cache_count >= 100:
                smallest_elements = sorted(self.avg_scores, key=lambda d: list(d.values())[0])[:10]
                self.select_index = [list(d.keys())[0] for d in smallest_elements]
                print(self.select_index)

                np.savetxt(self.config.pred_res_path, self.pred_res,
                           fmt=['%d', '%d', '%f', '%f', '%f', '%f', '%f'], delimiter=',')

                self.pred_res = []
                self.retrain_flag = True
                self.collect_flag = False
                self.cache_count = 0

    # send to cloud, get ground truth
    def get_cloud_target(self, frame):
        frame_request = message_transmission_pb2.FrameRequest(
            frame = frame,
            frame_shape=frame.shape,
        )
        try:
            channel = grpc.insecure_channel(self.config.server_ip)
            stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
            res = stub.frame_processor(frame_request)
        except Exception as e:
            logger.exception("the cloud can not reply, {}".format( e))
        else:
            logger.info(str(res))

    # retrain
    def retrain_worker(self):
        while self.retrain_flag:
            self.annotations = []
            for index in self.select_index:
                path = os.path.join(self.config.cache_path, index, '.jpg')
                frame = cv2.imread(path)
                target_res = self.get_cloud_target(frame)
                for score, label, box in zip(target_res['scores'], target_res['labels'], target_res['boxes']):
                    score = score.cpu().detach().item()
                    label = label.cpu().detach().item()
                    box = box.cpu().detach().tolist()
                    if score >= 0.60:
                        self.annotations.append(
                            (index, label, box[0], box[1],
                             box[2], box[3], score)
                        )
            if len(self.annotations):
                np.savetxt(self.config.annotation_path, self.annotations,
                           fmt=['%d', '%d', '%f', '%f', '%f', '%f', '%f'], delimiter=',')

            self.small_object_detection.retrain()
            self.collect_flag = True
            pass


    def start_edge_server(self):
        logger.info("edge {} server is starting".format(self.edge_id))
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        message_transmission_pb2_grpc.add_MessageTransmissionServicer_to_server(
            MessageTransmissionServicer(self.local_queue, self.edge_id, self.queue_info), server)
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
        else:
            logger.error('the policy does not exist.')

