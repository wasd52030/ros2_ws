import os
import tensorflow


def dynamicRangeQuantization() -> str:
    dynamic_tflite_path = "/home/sobel/ros2_ws/src/data/models/dynamicRangeQuantization.tflite"
    if not os.path.isfile(dynamic_tflite_path):
        print(tensorflow.__version__)
        
        # Load the TensorFlow model
        # The preprocessing and the post-processing steps should not be included in the TF Lite model graph
        # because some operations (ArgMax) might not support the delegates.
        # Insepct the graph using Netron https://lutzroeder.github.io/netron/
        converter = tensorflow.compat.v1.lite.TFLiteConverter.from_frozen_graph(
            graph_def_file="/home/sobel/ros2_ws/src/data/models/deeplabv3_mnv2_dm05_pascal_trainval/frozen_inference_graph.pb",
            input_arrays=["sub_7"],
            output_arrays=["ResizeBilinear_2"],
        )

        # Optional: Perform the simplest optimization known as post-training dynamic range quantization.
        # https://www.tensorflow.org/lite/performance/post_training_quantization#dynamic_range_quantization
        # You can refer to the same document for other types of optimizations.
        converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]

        with open(dynamic_tflite_path, "wb") as model:
            # Convert to TFLite Model
            tflite_model = converter.convert()
            tflite_model_size=model.write(tflite_model)
    else:
        tflite_model_size = os.path.getsize(dynamic_tflite_path)

    tf_model_size = os.path.getsize(
        "/home/sobel/ros2_ws/src/data/models/deeplabv3_mnv2_dm05_pascal_trainval/frozen_inference_graph.pb"
    )
    print("TensorFlow Model is  {} bytes".format(tf_model_size))
    print("TFLite Model is      {} bytes".format(tflite_model_size))
    print(
        "Post training dynamic range quantization saves {} bytes".format(
            tf_model_size - tflite_model_size
        )
    )

    return dynamic_tflite_path
