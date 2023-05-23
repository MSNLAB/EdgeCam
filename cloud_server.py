
import argparse

import threading
import time

import numpy as np
from queue import Queue
from datetime import datetime
import yaml
import munch
import grpc
from concurrent import futures

from loguru import logger
from database.database import DataBase
from edge.info import TASK_STATE
from grpc_server.rpc_server import MessageTransmissionServicer

from model_management.object_detection import Object_Detection
from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc


class CloudServer:
    def __init__(self, config):
        self.config = config
        self.server_id = config.server_id
        self.large_object_detection = Object_Detection(config, type='large inference')

        #create database and tables
        self.database = DataBase(self.config.database)
        self.database.use_database()
        self.database.create_table(self.config.edge_ids)

        # start the thread for local process
        self.local_queue = Queue(config.local_queue_maxsize)
        self.local_processor = threading.Thread(target=self.cloud_local, )
        self.local_processor.start()


    def cloud_local(self):
        while True:
            task = self.local_queue.get(block=True)
            task.set_frame_cloud(task.frame_edge)
            frame = task.frame_cloud
            high_boxes, high_class, high_score = self.large_object_detection.large_inference(frame)
            # scale the small result
            scale = task.raw_shape[0] / frame.shape[0]
            if high_boxes:
                high_boxes = (np.array(high_boxes) * scale).tolist()
                task.add_result(high_boxes, high_class, high_score)
            end_time = time.time()
            task.end_time = end_time
            # upload the result to database
            task.state = TASK_STATE.FINISHED
            self.update_table(task)
            self.ref_update(task)


    def ref_update(self, task):
        for ref_task in task.ref_list:
            detection_boxes, detection_class, detection_score = task.get_result()
            ref_task.add_result(detection_boxes, detection_class, detection_score)
            ref_task.state = TASK_STATE.FINISHED
            self.update_table(task)


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


    def start_server(self):
        logger.info("cloud server is starting")
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        message_transmission_pb2_grpc.add_MessageTransmissionServicer_to_server(
            MessageTransmissionServicer(self.local_queue, self.server_id), server)
        server.add_insecure_port('[::]:50051')
        server.start()
        logger.info("cloud server is listening")
        server.wait_for_termination()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="configuration description")
    parser.add_argument("--yaml_path", default="./config/config.yaml", help="input the path of *.yaml")
    args = parser.parse_args()
    with open(args.yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    # provide class-like access for dict
    config = munch.munchify(config)
    server_config = config.server
    cloud_server = CloudServer(server_config)
    cloud_server.start_server()