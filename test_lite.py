import cv2
import time
import helpers
import numpy as np
import tensorflow as tf


def predict(inp, out, input_details, output_details):
    test_img = cv2.imread(inp)
    x = helpers.get_image_array(inp, INPUT_SIZE, INPUT_SIZE)
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], np.array([x]))
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
interpreter = tf.lite.Interpreter(model_path=f"custom_model_{INPUT_SIZE}_new.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


predict("testing/test_img.png", "testing/result_lite.png", input_details, output_details)
predict("testing/test_img_2.png", "testing/result_2_lite.png", input_details, output_details)
predict("testing/test_img_3.png", "testing/result_3_lite.png", input_details, output_details)
predict("testing/test_img_4.png", "testing/result_4_lite.png", input_details, output_details)
predict("testing/test_img_5.png", "testing/result_5_lite.png", input_details, output_details)
predict("testing/test_img_6.png", "testing/result_6_lite.png", input_details, output_details)
