import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy
from datetime import datetime

lowerRed=numpy.array([0,90,128])
lowerRed=numpy.array([180,255,255])

class ImageSubscriber(Node):

    def __init__(self, node_name: str, *, context: rclpy.Context = None, cli_args: rclpy.List[str] = None, namespace: str = None, use_global_arguments: bool = True, enable_rosout: bool = True, start_parameter_services: bool = True, parameter_overrides: rclpy.List[rclpy.Parameter] = None, allow_undeclared_parameters: bool = False, automatically_declare_parameters_from_overrides: bool = False) -> None:
        super().__init__(node_name, context=context, cli_args=cli_args, namespace=namespace, use_global_arguments=use_global_arguments, enable_rosout=enable_rosout, start_parameter_services=start_parameter_services, parameter_overrides=parameter_overrides, allow_undeclared_parameters=allow_undeclared_parameters, automatically_declare_parameters_from_overrides=automatically_declare_parameters_from_overrides)

        self.sub=self.create_subscription(Image,'image_raw',self.listenerCallback,10)

        self.cvBridge=CvBridge()


    def imageProcess(self,image):
        cv2.imshow('ROS Webcam', image)

            # Press Enter to exit
        if cv2.waitKey(27) == 13:
            return


    def listenerCallback(self,data):
        self.get_logger().info(f"{datetime.now()}")

        image = self.cvBridge.imgmsg_to_cv2(data, 'bgr8')

        self.imageProcess(image)
        


def main(args=None):
    rclpy.init(args=args)
    node=ImageSubscriber("webcamSubscriber")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()