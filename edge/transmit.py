import os

import cv2
import grpc
from loguru import logger

from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc
from tools.convert_tool import cv2_to_base64



# send to cloud, get ground truth
def get_cloud_target(server_ip, select_index, cache_path):

    def requset_stream():
        for index in select_index:
            path = os.path.join(cache_path, 'frames', '{}.jpg'.format(index))
            logger.debug(path)
            frame = cv2.imread(path)
            encoded_image = cv2_to_base64(frame)
            frame_request = message_transmission_pb2.FrameRequest(
                frame=encoded_image,
                frame_shape=str(frame.shape),
                frame_index= index,
            )
            yield frame_request

    try:
        channel = grpc.insecure_channel(server_ip)
        stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
        res = stub.frame_processor(requset_stream())
        result_dict = eval(res.response)
    except Exception as e:
        logger.exception("the cloud can not reply, {}".format(e))
    else:
        logger.debug("res{}".format(result_dict))
    return result_dict



import socket

def is_network_connected(address):
    ip, port = address.split(':')[0], int(address.split(':')[1])
    try:
        socket.create_connection((ip, port), timeout=1)
        return True
    except OSError:
        return False
