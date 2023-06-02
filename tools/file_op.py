import os
from loguru import logger

def creat_folder(folder_path):
    frames_path = os.path.join(folder_path, 'frames')
    logger.debug("creat {}".format(frames_path))
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)


def clear_folder(folder_path):
    logger.debug("clear floder")

    if not os.path.exists(folder_path):
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.debug(f"Failed to delete file: {file_path}. Reason: {e}")

    frames_path = os.path.join(folder_path, 'frames')
    if not os.path.exists(frames_path):
        return
    for filename in os.listdir(frames_path):
        file_path = os.path.join(frames_path, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.debug(f"Failed to delete file: {file_path}. Reason: {e}")



def sample_files(root,indexs):
    logger.debug("clear index {}".format(indexs))
    for filename in os.listdir(root):
        if int(filename.split('.')[0]) not in indexs:
            file_path = os.path.join(root, filename)
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"remove error: {e}")