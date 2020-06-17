import cv2
import time
import helpers
import numpy as np
import tensorflow as tf
from LR_ASPP import LiteRASSP


# latest_weights = tf.train.latest_checkpoint(f"checkpoints_{INPUT_SIZE}")
model = LiteRASSP(input_shape=(1024, 1024, 3), n_class=2, alpha=1.0, weights=None).build(plot=True)
# model.load_weights(latest_weights)
model.summary()


def predict(inp, out):
    test_img = cv2.imread(inp)
    x = helpers.get_image_array(inp, 1024, 1024)
    start_time = time.time()
    result = model.predict(np.array([x]))[0]
    print("--- %s seconds ---" % (time.time() - start_time))
    print(model.output_shape)
    result = result.reshape((1024,  1024, 2)).argmax(axis=2)
    pw, ph, _ = test_img.shape
    seg_img = helpers.visualize_segmentation(result, test_img, n_classes=2, prediction_width=ph, prediction_height=pw)

    cv2.imwrite(out, seg_img)



predict("testing/test_img.png", "testing/result_small.png")
predict("testing/test_img_2.png", "testing/result_2_small.png")
# predict("testing/test_img_3.png", "testing/result_3.png")
# predict("testing/test_img_4.png", "testing/result_4.png")
# predict("testing/test_img_5.png", "testing/result_5.png")
# predict("testing/test_img_6.png", "testing/result_6.png")
