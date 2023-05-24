import os
from loguru import logger

def creat_folder(folder_path):
    logger.debug("creat {}".format(folder_path))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def clear_folder(folder_path):
    # 确保文件夹存在
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



