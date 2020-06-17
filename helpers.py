import cv2
import numpy as np

INPUT_SIZE = 1024
CLASS_COLORS = [(0, 0, 0), (255, 255, 255)]


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
        seg_img[:, :, 0] += ((seg_arr_c)*(CLASS_COLORS[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(CLASS_COLORS[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(CLASS_COLORS[c][2])).astype('uint8')

    return seg_img


def visualize_segmentation(seg_arr, inp_img=None, n_classes=None,
                           prediction_width=None, prediction_height=None):
    if n_classes is None:
        n_classes = np.max(seg_arr)

    seg_img = get_colored_segmentation_image(seg_arr, n_classes)

    if inp_img is not None:
        orininal_h = inp_img.shape[0]
        orininal_w = inp_img.shape[1]
        seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    if (prediction_height is not None) and (prediction_width is not None):
        seg_img = cv2.resize(seg_img, (prediction_width, prediction_height))
        if inp_img is not None:
            inp_img = cv2.resize(inp_img,
                                 (prediction_width, prediction_height))

    seg_img = overlay_seg_image(inp_img, seg_img)

    return seg_img