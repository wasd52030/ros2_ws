import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy
from datetime import datetime
import os
import sys

class faceDetector(Node):
    def __init__(
        self,
        node_name: str,
        *,
        context: rclpy.Context = None,
        cli_args: rclpy.List[str] = None,
        namespace: str = None,
        use_global_arguments: bool = True,
        enable_rosout: bool = True,
        start_parameter_services: bool = True,
        parameter_overrides: rclpy.List[rclpy.Parameter] = None,
        allow_undeclared_parameters: bool = False,
        automatically_declare_parameters_from_overrides: bool = False,
    ) -> None:
        super().__init__(
            node_name,
            context=context,
            cli_args=cli_args,
            namespace=namespace,
            use_global_arguments=use_global_arguments,
            enable_rosout=enable_rosout,
            start_parameter_services=start_parameter_services,
            parameter_overrides=parameter_overrides,
            allow_undeclared_parameters=allow_undeclared_parameters,
            automatically_declare_parameters_from_overrides=automatically_declare_parameters_from_overrides,
        )

        self.sub = self.create_subscription(
            Image, "image_raw", self.listenerCallback, 10
        )

        self.cvBridge = CvBridge()

        # 引入人像識別訓練庫 haarcascade_frontalface_default.xml
        # https://github.com/opencv/opencv/tree/master/data/haarcascades
        self.face_patterns = cv2.CascadeClassifier(
            "/home/sobel/ros2_ws/src/data/models/haarcascade_frontalface_default.xml"
        )
        self.eyes_patterns = cv2.CascadeClassifier("/home/sobel/ros2_ws/src/data/models/haarcascade_eye.xml")
        self.mouth_patterns = cv2.CascadeClassifier("/home/sobel/ros2_ws/src/data/models/haarcascade_mcs_mouth.xml")

    def imageProcess(self, image: cv2.typing.MatLike):
        image = cv2.flip(image, 1)  # 1:左右鏡像，0:上下左右顛倒，-1:上下顛倒

        # 獲取識別到的人臉                      (偵測的圖,每次放大多少去偵測,閥值預設3,從多少開始偵測50*1.15=57.5
        faces = self.face_patterns.detectMultiScale(
            image,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(50, 50),
            maxSize=(500, 500),
        )

        for x, y, w, h in faces:  # 顏色  bgr  寬度
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)

            # 口罩辨識
            # 找到臉部後，找臉的下半部，檢查這區域是否有嘴巴存在
            mouths = self.mouth_patterns.detectMultiScale(
                image[y : y + y // 2 +h//2, x : x + w],
                scaleFactor=1.5,
                minNeighbors=10,
                minSize=(50, 50),
                maxSize=(150, 150),
            )
            if len(mouths) <= 0:
                cv2.rectangle(
                    image, (x, y + y // 2), (x + w, y + y // 2 + h // 2), (255, 0, 0), 5
                )
                

            eyes = self.eyes_patterns.detectMultiScale(
                image[y : y + h, x : x + w],
                scaleFactor=1.15,
                minNeighbors=3 if len(mouths) <= 0 else 5,
                minSize=(5, 5),
                maxSize=(100, 100),
            )

            # print(x, y)
            # print((x, y + y // 2), (x + w, y + y // 2 + h // 2))
            # cv2.rectangle(
            #     image, (x, y + y // 2), (x + w, y + y // 2 + h // 2), (255, 0, 0), 5
            # )

            for x1, y1, w1, h1 in eyes:  # 顏色       寬度
                cv2.rectangle(
                    image, (x1 + x, y1 + y), (x + x1 + w1, y + y1 + h1), (0, 0, 255), 5
                )

            # for x1, y1, w1, h1 in mouths:  # 顏色       寬度
            #     cv2.rectangle(
            #         image, (x1 + x, y1 + y), (x + x1 + w1, y + y1 + h1), (0, 0, 255), 5
            #     )

        cv2.imshow("ROS Webcam", image)

        # Press Enter to exit
        if cv2.waitKey(27) == 27:
            return

    def listenerCallback(self, data):
        self.get_logger().info(f"{datetime.now()}")

        image = self.cvBridge.imgmsg_to_cv2(data, "bgr8")

        self.imageProcess(image)


def main(args=None):
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    rclpy.init(args=args)
    node = faceDetector("webcamSubscriber")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
