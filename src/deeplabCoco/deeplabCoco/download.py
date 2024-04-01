import os
import subprocess

if not os.path.isdir("models"):
    os.mkdir("models")

MODEL_NAME = "mobilenetv2_dm05_coco_voc_trainval"  # @param ['mobilenetv2_dm05_coco_voc_trainval', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainval']

DOWNLOAD_URL_PREFIX = "http://download.tensorflow.org/models/"
MODEL_URLS = {
    "mobilenetv2_dm05_coco_voc_trainval": "deeplabv3_mnv2_dm05_pascal_trainval_2018_10_01.tar.gz",
    "mobilenetv2_coco_voctrainval": "deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz",
    "xception_coco_voctrainval": "deeplabv3_pascal_trainval_2018_01_04.tar.gz",
}

MODEL_TAR = MODEL_URLS[MODEL_NAME]
MODEL_URL = DOWNLOAD_URL_PREFIX + MODEL_TAR
subprocess.run(["wget", "-O", f"/home/sobel/ros2_ws/src/data/models/{MODEL_TAR}", f"{MODEL_URL}"])

subprocess.run(["tar", "zxvf", f"/home/sobel/ros2_ws/src/data/models/{MODEL_TAR}", "-C", "/home/sobel/ros2_ws/src/data/models"])