import os
import tensorflow


def float16Quantization() -> str:
    f16_tflite_path = "/home/sobel/ros2_ws/src/data/models/float16Quantization.tflite"
    if not os.path.isfile(f16_tflite_path):
        converter = tensorflow.compat.v1.lite.TFLiteConverter.from_frozen_graph(
            graph_def_file="/home/sobel/ros2_ws/src/data/models/deeplabv3_mnv2_dm05_pascal_trainval/frozen_inference_graph.pb",
            input_arrays=["sub_7"],
            output_arrays=["ResizeBilinear_2"],
        )
        converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tensorflow.float16]

        with open(f16_tflite_path, "wb") as model:
            # Convert to TFLite Model
            tflite_model = converter.convert()
            tflite_model_size = model.write(tflite_model)
    else:
        tflite_model_size = os.path.getsize(f16_tflite_path)

    tf_model_size = os.path.getsize(
        "/home/sobel/ros2_ws/src/data/models/deeplabv3_mnv2_dm05_pascal_trainval/frozen_inference_graph.pb"
    )
    print("TensorFlow Model is  {} bytes".format(tf_model_size))
    print("TFLite Model is      {} bytes".format(tflite_model_size))
    print(
        "Post training float16 quantization saves {} bytes".format(
            tf_model_size - tflite_model_size
        )
    )

    return f16_tflite_path
