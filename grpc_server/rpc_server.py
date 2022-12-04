import argparse
import yaml
import munch
import grpc
from concurrent import futures

from loguru import logger

from tools.convert_tool import base64_to_cv2
from grpc_server import message_transfer_pb2, message_transfer_pb2_grpc
from inferencer.object_detection import Object_Detection


class MessageTransferServer(message_transfer_pb2_grpc.MessageTransferServicer):
    def __init__(self, config):
        self.large_object_detection = Object_Detection(config, type='large inference')

    def image_processor(self, request, context):
        base64_frame = request.frame
        frame_shape = tuple(int(s) for s in request.frame_shape[1:-1].split(","))
        frame = base64_to_cv2(base64_frame).reshape(frame_shape)
        high_boxes, high_class, high_score = self.large_object_detection.large_inference(frame)
        result_dict = {
            'boxes': high_boxes,
            'class': high_class,
            'score': high_score,
        }
        reply = message_transfer_pb2.MessageReply(
            result=str(result_dict),
            frame_shape="",
        )
        return reply


def start_server(config):
    logger.info("grpc server is starting")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    message_transfer_pb2_grpc.add_MessageTransferServicer_to_server(MessageTransferServer(config), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("grpc server is listening")
    server.wait_for_termination()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="configuration description")
    parser.add_argument("--yaml_path", default="../config/config.yaml", help="input the path of *.yaml")
    args = parser.parse_args()
    with open(args.yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    # provide class-like access for dict
    config = munch.munchify(config)
    server_config = config.server

    start_server(server_config)