import numpy as np
import tensorflow as tf
from LR_ASPP import LiteRASSP

N_CHANNELS = 1
INPUT_SIZE = 1024
latest_weights = tf.train.latest_checkpoint(f"checkpoints/{INPUT_SIZE}_gs_pd_1298_158")
print(latest_weights)
model = LiteRASSP(input_shape=(INPUT_SIZE, INPUT_SIZE, N_CHANNELS), n_class=2, alpha=1.0, weights=None).build(plot=True)
model.load_weights(latest_weights)

# Load the MobileNet tf.keras model.
# model = tf.keras.applications.MobileNetV2(
#     weights="imagenet", input_shape=(224, 224, 3))

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.inference_type = tf.compat.v1.lite.constants.QUANTIZED_UINT8
# converter.optimization = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                        tf.lite.OpsSet.SELECT_TF_OPS,]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8  # or tf.uint8
# converter.inference_output_type = tf.int8  # or tf.uint8
# converter.allow_custom_ops = True
# converter.experimental_new_converter = True
tflite_model = converter.convert()

with tf.io.gfile.GFile(f"tflite_models/custom_model_{INPUT_SIZE}_gs_pd_1298_158.tflite", "wb") as f:
    f.write(tflite_model)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the TensorFlow Lite model on random input data.
input_shape = input_details[0]["shape"]
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]["index"], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
tflite_results = interpreter.get_tensor(output_details[0]["index"])

# Test the TensorFlow model on random input data.
tf_results = model(tf.constant(input_data))

# Compare the result.
for tf_result, tflite_result in zip(tf_results, tflite_results):
    np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)
