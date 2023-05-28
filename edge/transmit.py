import grpc
from loguru import logger

from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc
from tools.convert_tool import cv2_to_base64


# send to cloud, get ground truth
def get_cloud_target(server_ip, frame):
    encoded_image = cv2_to_base64(frame)
    frame_request = message_transmission_pb2.FrameRequest(
        frame=encoded_image,
        frame_shape=str(frame.shape),
    )
    try:
        channel = grpc.insecure_channel(server_ip)
        stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
        res = stub.frame_processor(frame_request)
        result_dict = eval(res.response)
        logger.debug("res{}".format(result_dict))
    except Exception as e:
        logger.exception("the cloud can not reply, {}".format(e))
    else:
        logger.info(str(res))
    return result_dict