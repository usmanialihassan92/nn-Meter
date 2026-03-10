import tensorflow as tf
import tf2onnx


def export_tf_model(output_path="onnx/mobilenetv3_tf.onnx"):

    # Load pretrained model
    model = tf.keras.applications.MobileNetV3Small(weights="imagenet")

    # Input signature
    input_signature = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="input"),)

    print("Exporting TensorFlow model to ONNX...")

    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13,
        output_path=output_path,
    )

    print(f"ONNX model saved to {output_path}")


if __name__ == "__main__":
    export_tf_model()