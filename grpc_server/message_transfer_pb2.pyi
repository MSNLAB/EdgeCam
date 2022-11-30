from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class MessageReply(_message.Message):
    __slots__ = ["frame_shape", "result"]
    FRAME_SHAPE_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    frame_shape: str
    result: str
    def __init__(self, result: _Optional[str] = ..., frame_shape: _Optional[str] = ...) -> None: ...

class MessageRequest(_message.Message):
    __slots__ = ["frame", "frame_shape"]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    FRAME_SHAPE_FIELD_NUMBER: _ClassVar[int]
    frame: str
    frame_shape: str
    def __init__(self, frame: _Optional[str] = ..., frame_shape: _Optional[str] = ...) -> None: ...
