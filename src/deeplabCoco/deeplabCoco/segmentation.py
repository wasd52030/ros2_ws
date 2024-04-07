from datetime import datetime
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy
from numpy.typing import ArrayLike
from PIL import Image as PILImage, ImageOps
from matplotlib import pyplot
import tensorflow

from .dynamicRangeQuantization import dynamicRangeQuantization
from .float16Quantization import float16Quantization
from .int8Quantization import int8Quantization
import matplotlib


class deeplavCocoSegmentation(Node):
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

        pyplot.figure(figsize=(6.4, 6.4))

        self.model_dict = {
            "dynamic-range": dynamicRangeQuantization(),
            "fp16": float16Quantization(),
            "int8": int8Quantization(),
        }

        self.tensorflowlite_model_type = (
            "fp16"  # @param ['dynamic-range', 'fp16', 'int8']
        )

        # Load the model.
        interpreter = tensorflow.lite.Interpreter(
            model_path=self.model_dict[self.tensorflowlite_model_type]
        )

        # Set model input.
        self.input_details = interpreter.get_input_details()
        interpreter.allocate_tensors()

        # Get image size - converting from BHWC to WH
        self.input_size = (
            self.input_details[0]["shape"][2],
            self.input_details[0]["shape"][1],
        )

        pyplot.ion()

    def create_pascal_label_colormap(self):
        """Creates a label colormap used in PASCAL VOC segmentation benchmark.

        Returns:
        A Colormap for visualizing segmentation results.
        """
        colormap = numpy.zeros((256, 3), dtype=int)
        ind = numpy.arange(256, dtype=int)

        for shift in reversed(range(8)):
            for channel in range(3):
                colormap[:, channel] |= ((ind >> channel) & 1) << shift
            ind >>= 3

        return colormap

    def label_to_color_image(self, label: ArrayLike):
        """Adds color defined by the dataset colormap to the label.

        Args:
        label: A 2D array with integer type, storing the segmentation label.

        Returns:
        result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the PASCAL color map.

        Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.
        """
        if label.ndim != 2:
            raise ValueError("Expect 2-D input label")

        colormap = self.create_pascal_label_colormap()

        if numpy.max(label) >= len(colormap):
            raise ValueError("label value too large.")

        return colormap[label]

    def vis_segmentation(self, image: PILImage, seg_map: ArrayLike):
        pyplot.clf()

        LABEL_NAMES = numpy.asarray(
            [
                "background",
                "aeroplane",
                "bicycle",
                "bird",
                "boat",
                "bottle",
                "bus",
                "car",
                "cat",
                "chair",
                "cow",
                "diningtable",
                "dog",
                "horse",
                "motorbike",
                "person",
                "pottedplant",
                "sheep",
                "sofa",
                "train",
                "tv",
            ]
        )

        FULL_LABEL_MAP = numpy.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
        FULL_COLOR_MAP = self.label_to_color_image(FULL_LABEL_MAP)

        seg_image = self.label_to_color_image(seg_map).astype(numpy.uint8)

        """Visualizes input image, segmentation map and overlay view."""
        pyplot.imshow(image)
        pyplot.imshow(seg_image, alpha=0.7)
        pyplot.axis("off")

        pyplot.draw()
        pyplot.pause(0.01)

        # return seg_image

    def imageProcess(self, cam_img: cv2.typing.MatLike):
        image = PILImage.fromarray(cam_img)
        old_size = image.size  # old_size is in (width, height) format
        desired_ratio = self.input_size[0] / self.input_size[1]
        old_ratio = old_size[0] / old_size[1]

        if old_ratio < desired_ratio:  # '<': cropping, '>': padding
            new_size = (old_size[0], int(old_size[0] / desired_ratio))
        else:
            new_size = (int(old_size[1] * desired_ratio), old_size[1])

        print(new_size, old_size)

        # Cropping the original image to the desired aspect ratio
        delta_w = new_size[0] - old_size[0]
        delta_h = new_size[1] - old_size[1]
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - (delta_w // 2),
            delta_h - (delta_h // 2),
        )
        cropped_image = ImageOps.expand(image, padding)

        # Resize the cropped image to the desired model size
        resized_image = cropped_image.convert("RGB").resize(
            self.input_size, PILImage.BILINEAR
        )

        # Convert to a NumPy array, add a batch dimension, and normalize the image.
        image_for_prediction = numpy.asarray(resized_image).astype(numpy.float32)
        image_for_prediction = numpy.expand_dims(image_for_prediction, 0)
        image_for_prediction = image_for_prediction / 127.5 - 1

        # Load the model.
        interpreter = tensorflow.lite.Interpreter(
            model_path=self.model_dict[self.tensorflowlite_model_type]
        )

        # Invoke the interpreter to run inference.
        interpreter.allocate_tensors()
        interpreter.set_tensor(self.input_details[0]["index"], image_for_prediction)
        interpreter.invoke()

        # Retrieve the raw output map.
        raw_prediction = interpreter.tensor(
            interpreter.get_output_details()[0]["index"]
        )()

        # Post-processing: convert raw output to segmentation output
        ## Method 1: argmax before resize - this is used in some frozen graph
        # seg_map = numpy.squeeze(numpy.argmax(raw_prediction, axis=3)).astype(numpy.int8)
        # seg_map = numpy.asarray(Image.fromarray(seg_map).resize(image.size, resample=Image.NEAREST))
        ## Method 2: resize then argmax - this is used in some other frozen graph and produce smoother output
        width, height = cropped_image.size
        seg_map = tensorflow.argmax(
            tensorflow.image.resize(raw_prediction, (height, width)), axis=3
        )
        seg_map = tensorflow.squeeze(seg_map).numpy().astype(numpy.int8)

        self.vis_segmentation(cropped_image, seg_map)

    def listenerCallback(self, data: Image):
        self.get_logger().info(f"{datetime.now()}")

        image = self.cvBridge.imgmsg_to_cv2(data, "bgr8")

        self.imageProcess(image)


def main(args=None):
    matplotlib.use("tkAgg")
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # 初始化ROS
    rclpy.init(args=args)

    # 創建Node
    node = deeplavCocoSegmentation("deeplabCoco")

    # 讓Node持續運行
    rclpy.spin(node)

    # 關閉ROS
    rclpy.shutdown()


if __name__ == "__main__":
    main()


