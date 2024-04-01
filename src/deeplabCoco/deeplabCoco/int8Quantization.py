import os
import tensorflow


def representative_dataset_gen():
    for _ in range(10):
        dummy_image = tensorflow.random.uniform(
            [1, 513, 513, 3], 0.0, 255.0, dtype=tensorflow.float32
        )
        dummy_image = dummy_image / 127.5 - 1
        yield [dummy_image]


def int8Quantization() -> str:
    int_tflite_path = "/home/sobel/ros2_ws/src/data/models/int8Quantization.tflite"
    if not os.path.isfile(int_tflite_path):
        converter = tensorflow.compat.v1.lite.TFLiteConverter.from_frozen_graph(
            graph_def_file="/home/sobel/ros2_ws/src/data/models/deeplabv3_mnv2_dm05_pascal_trainval/frozen_inference_graph.pb",
            input_arrays=["sub_7"],
            output_arrays=["ResizeBilinear_2"],
        )
        converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen

        tflite_model = converter.convert()

        with open(int_tflite_path, "wb") as model:
            # Convert to TFLite Model
            tflite_model = converter.convert()
            tflite_model_size = model.write(tflite_model)
    else:
        tflite_model_size = os.path.getsize(int_tflite_path)

    tf_model_size = os.path.getsize(
        "/home/sobel/ros2_ws/src/data/models/deeplabv3_mnv2_dm05_pascal_trainval/frozen_inference_graph.pb"
    )
    print("TensorFlow Model is  {} bytes".format(tf_model_size))
    print("TFLite Model is      {} bytes".format(tflite_model_size))
    print(
        "Post training int8 quantization saves {} bytes".format(
            tf_model_size - tflite_model_size
        )
    )
