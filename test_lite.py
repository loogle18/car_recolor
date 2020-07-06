import cv2
import time
import helpers
import numpy as np
import tensorflow as tf


N_CHANNELS = 1
INPUT_SIZE = 1024


def predict(inp, out, input_details, output_details):
    test_img = cv2.imread(inp, 1)
    x = helpers.get_image_array(inp, INPUT_SIZE, INPUT_SIZE)
    # print(x)
    xx = np.array([x]).reshape(1, INPUT_SIZE, INPUT_SIZE, N_CHANNELS)
    print(xx.shape)
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], xx)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]['index'])
    # print(output_data)
    print("--- %s seconds ---" % (time.time() - start_time))
    # print(model.output_shape)
    result = result.reshape((INPUT_SIZE,  INPUT_SIZE, 2)).argmax(axis=2)
    pw, ph, _ = test_img.shape
    seg_img = helpers.visualize_segmentation(result, test_img, n_classes=2, prediction_width=ph, prediction_height=pw)

    cv2.imwrite(out, seg_img)


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=f"tflite_models/custom_model_{INPUT_SIZE}_gs_pd_1298_158.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
print(input_details)
output_details = interpreter.get_output_details()
print(output_details)
