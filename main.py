import cv2
import time
import numpy as np
import tensorflow as tf
from LR_ASPP import LiteRASSP

import keras.backend as K


latest_weights = tf.train.latest_checkpoint("checkpoints")
print(latest_weights)
model = LiteRASSP(input_shape=(1024, 1024, 3), n_class=2, alpha=1.0, weights=None).build(plot=True)
model.load_weights(latest_weights)
# print(model)
# model.summary()


def get_image_array(image_input, width, height):
    img = cv2.imread(image_input, 1)
    img = cv2.resize(img, (width, height))
    img = img.astype(np.float32)
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    img = img[:, :, ::-1]
    return img

orig_img = cv2.imread("test_img.png")
img = get_image_array("test_img.png", 1024, 1024)



def resize_image(x, s):
    return K.resize_images(x,height_factor=s[0],width_factor=s[1],data_format="channels_last",interpolation='bilinear')



class_colors = [(0, 0, 0), (255, 255, 255)]


def get_image_array(image_input, width, height):
    img = cv2.imread(image_input, 1)
    img = cv2.resize(img, (width, height))
    img = img.astype(np.float32)
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    img = img[:, :, ::-1]
    return img


def overlay_seg_image(inp_img, seg_img):
    orininal_h = inp_img.shape[0]
    orininal_w = inp_img.shape[1]
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))
    # added_image = cv2.addWeighted(inp_img,0,seg_img,0.1,0)
    # cv2.imshow('image',added_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    seg_img = seg_img * 0.5
    green = np.ones(inp_img.shape, dtype=np.float)*(255,255,255)
    out = green*seg_img + inp_img*(1.0-seg_img)
    fused_img = (inp_img + seg_img).astype('uint8')
    return out


def get_colored_segmentation_image(seg_arr, n_classes):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c)*(class_colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(class_colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(class_colors[c][2])).astype('uint8')

    return seg_img


def visualize_segmentation(seg_arr, inp_img=None, n_classes=None,
                           prediction_width=None, prediction_height=None):
    if n_classes is None:
        n_classes = np.max(seg_arr)

    seg_img = get_colored_segmentation_image(seg_arr, n_classes)
    # seg_img2 = cv2.cvtColor(seg_img, cv2.COLOR_BGR2BGRA)
    # cv2.imwrite("out_fname.png", seg_img)
    # seg_img = cv2.imread('out_fname.png', cv2.IMREAD_UNCHANGED)
    # print(seg_img.shape)
    #make mask of where the transparent bits are
    # trans_mask = seg_img[:,:,3] == 0

    # #replace areas of transparency with white and not transparent
    # seg_img[trans_mask] = [255, 255, 255, 255]

    # #new image without alpha channel...
    # seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGRA2BGR)
    # *_, alpha = cv2.split(seg_img)
    # im = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    # seg_img = cv2.merge((im, im, im, alpha))
    
    # cv2.imwrite("out_fname2.png", seg_img)   
    if inp_img is not None:
        orininal_h = inp_img.shape[0]
        orininal_w = inp_img.shape[1]
        seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    if (prediction_height is not None) and (prediction_width is not None):
        seg_img = cv2.resize(seg_img, (prediction_width, prediction_height))
        if inp_img is not None:
            inp_img = cv2.resize(inp_img,
                                 (prediction_width, prediction_height))

    # cv2.imwrite("seg_img.png", seg_img)
    seg_img = overlay_seg_image(inp_img, seg_img)

    return seg_img



# test_img = cv2.imread("test_img.png")
# test_img = test_img[:,:,::-1]
# test_img = cv2.resize(test_img, (1024,1024))
# tensor = tf.convert_to_tensor(test_img)
# tensor = tf.dtypes.cast(tensor, 'float32')
# test_tensor= tf.divide(tensor, 255.0)
# test_tensor = tf.expand_dims(test_tensor, 0)


# test_img = cv2.imread("test_img.png")
# x = get_image_array("test_img.png", 1024, 1024)
# result = model.predict(np.array([x]))[0]
# print(model.output_shape)
# result = result.reshape((1024,  1024, 2)).argmax(axis=2)
# pw, ph, _ = test_img.shape
# seg_img = visualize_segmentation(result, test_img, n_classes=2, prediction_width=ph, prediction_height=pw)
# cv2.imwrite("result.png", seg_img)



# result = result.reshape((1024,  1024, 3)).argmax(axis=2)

# (orininal_w, orininal_h, _) = orig_img.shape
# new_img = cv2.resize(result, (orininal_w, orininal_h))

# new_img.imwrite("result.png", new_img)

# new_img = np.squeeze(result)
# strides = [16, 16]
# new_img = resize_image(result, strides)
# new_img = np.squeeze(new_img)
# print(new_img.shape)
# new_img = [new_img, new_img, new_img]
# new_img = np.transpose(new_img, (1, 2, 0))
# new_img.imwrite("result.png", new_img)


# print(result)





def predict(inp, out):
    test_img = cv2.imread(inp)
    x = get_image_array(inp, 1024, 1024)
    start_time = time.time()
    result = model.predict(np.array([x]))[0]
    print("--- %s seconds ---" % (time.time() - start_time))
    print(model.output_shape)
    result = result.reshape((1024,  1024, 2)).argmax(axis=2)
    pw, ph, _ = test_img.shape
    seg_img = visualize_segmentation(result, test_img, n_classes=2, prediction_width=ph, prediction_height=pw)

    cv2.imwrite(out, seg_img)



predict("test_img.png", "result.png")
predict("test_img_2.png", "result_2.png")
predict("test_img_3.png", "result_3.png")
predict("test_img_4.png", "result_4.png")
predict("test_img_5.png", "result_5.png")
predict("test_img_6.png", "result_6.png")
