from edge.task import Task
from tools.convert_tool import base64_to_cv2
from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc


class MessageTransmissionServicer(message_transmission_pb2_grpc.MessageTransmissionServicer):
    def __init__(self, local_queue, id, queue_info=None):
        self.local_queue = local_queue
        self.id = id
        self.queue_info = queue_info

    def task_processor(self, request, context):

        base64_frame = request.frame
        frame_shape = tuple(int(s) for s in request.new_shape[1:-1].split(","))
        frame = base64_to_cv2(base64_frame).reshape(frame_shape)
        raw_shape = tuple(int(s) for s in request.raw_shape[1:-1].split(","))

        task = Task(request.source_edge_id, request.frame_index, frame, float(request.start_time), raw_shape)
        task.other = True
        if request.part_result != "":
            part_result = eval(request.part_result)
            task.add_result(part_result['boxes'], part_result['labels'], part_result['scores'])
        if request.note == "edge process":
            task.edge_process = True
        self.local_queue.put(task, block=True)

        reply = message_transmission_pb2.MessageReply(
            destination_id=self.id,
            local_length=self.local_queue.qsize(),
            response='offload to {} successfully'.format(self.id)
        )

        return reply

    def get_queue_info(self, request, context):
        self.queue_info['{}'.format(request.source_edge_id)] = request.local_length
        reply = message_transmission_pb2.InfoReply(
            destination_id=self.id,
            local_length=self.local_queue.qsize(),
        )
        return reply


