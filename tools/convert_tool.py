import os
import cv2
import base64
import numpy as np
from sys import getsizeof


def cv2_to_base64(frame, qp):
    """
    covert image nparray to stream data
    :param frame:
    :param qp:
    :return:
    """
    encode = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, qp])[1]
    encode_str = encode.tobytes()
    base64_str = base64.b64encode(encode_str)
    return base64_str


def base64_to_cv2(string):
    """
    covert the stream data to image
    :param string:
    :return:
    """
    imgString = base64.b64decode(string)
    nparr = np.frombuffer(imgString, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


if __name__ == '__main__':
    img = cv2.imread('./inferencer/test_frame/1.jpg')
    print(img.shape)
    mem = os.path.getsize('../inferencer/test_frame/1.jpg')
    base64str = cv2_to_base64(img, 100)
    new_image = base64_to_cv2(base64str).reshape(img.shape)
    print(new_image.shape)
    cv2.imwrite('./inferencer/test_frame/1_ende.jpg', new_image)