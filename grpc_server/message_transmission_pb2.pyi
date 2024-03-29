from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class FrameReply(_message.Message):
    __slots__ = ["frame_shape", "response"]
    FRAME_SHAPE_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    frame_shape: str
    response: str
    def __init__(self, response: _Optional[str] = ..., frame_shape: _Optional[str] = ...) -> None: ...

class FrameRequest(_message.Message):
    __slots__ = ["frame", "frame_index", "frame_shape"]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    FRAME_INDEX_FIELD_NUMBER: _ClassVar[int]
    FRAME_SHAPE_FIELD_NUMBER: _ClassVar[int]
    frame: str
    frame_index: int
    frame_shape: str
    def __init__(self, frame: _Optional[str] = ..., frame_shape: _Optional[str] = ..., frame_index: _Optional[int] = ...) -> None: ...

class InfoReply(_message.Message):
    __slots__ = ["destination_id", "local_length"]
    DESTINATION_ID_FIELD_NUMBER: _ClassVar[int]
    LOCAL_LENGTH_FIELD_NUMBER: _ClassVar[int]
    destination_id: int
    local_length: int
    def __init__(self, destination_id: _Optional[int] = ..., local_length: _Optional[int] = ...) -> None: ...

class InfoRequest(_message.Message):
    __slots__ = ["local_length", "source_edge_id"]
    LOCAL_LENGTH_FIELD_NUMBER: _ClassVar[int]
    SOURCE_EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    local_length: int
    source_edge_id: int
    def __init__(self, source_edge_id: _Optional[int] = ..., local_length: _Optional[int] = ...) -> None: ...

class MessageReply(_message.Message):
    __slots__ = ["destination_id", "local_length", "response"]
    DESTINATION_ID_FIELD_NUMBER: _ClassVar[int]
    LOCAL_LENGTH_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    destination_id: int
    local_length: int
    response: str
    def __init__(self, destination_id: _Optional[int] = ..., local_length: _Optional[int] = ..., response: _Optional[str] = ...) -> None: ...

class MessageRequest(_message.Message):
    __slots__ = ["frame", "frame_index", "new_shape", "note", "part_result", "raw_shape", "ref_list", "source_edge_id", "start_time"]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    FRAME_INDEX_FIELD_NUMBER: _ClassVar[int]
    NEW_SHAPE_FIELD_NUMBER: _ClassVar[int]
    NOTE_FIELD_NUMBER: _ClassVar[int]
    PART_RESULT_FIELD_NUMBER: _ClassVar[int]
    RAW_SHAPE_FIELD_NUMBER: _ClassVar[int]
    REF_LIST_FIELD_NUMBER: _ClassVar[int]
    SOURCE_EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    frame: str
    frame_index: int
    new_shape: str
    note: str
    part_result: str
    raw_shape: str
    ref_list: str
    source_edge_id: int
    start_time: str
    def __init__(self, source_edge_id: _Optional[int] = ..., frame_index: _Optional[int] = ..., start_time: _Optional[str] = ..., frame: _Optional[str] = ..., part_result: _Optional[str] = ..., raw_shape: _Optional[str] = ..., new_shape: _Optional[str] = ..., ref_list: _Optional[str] = ..., note: _Optional[str] = ...) -> None: ...
