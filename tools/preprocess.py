import os
import cv2


def frame_resize(frame, new_height):
    """
    change the size of video frame
    :param frame:
    :param new_height:
    :return:
    """
    width = frame.shape[1]
    hight = frame.shape[0]
    scale = new_height / hight
    new_width = width * scale
    new_image = cv2.resize(frame, (int(new_width), int(new_height)), interpolation=cv2.INTER_AREA)
    return new_image


def frame_change_quality(frame, qp):
    """
    change the quality of video frame
    :param frame:
    :param qp:
    :return:
    """
    change_quality_buffer = './quality_buffer'
    frame_path = os.path.join(change_quality_buffer, 'temp.jpg')
    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, qp])
    changed_image = cv2.imread(frame_path)
    return changed_image


if __name__ == '__main__':
    pass