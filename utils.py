import os
import cv2
import glob
import random
import itertools
import numpy as np
from tqdm import tqdm
from stuff.augmentation import augment_seg


IMAGE_NORM = "sub_mean"
IMAGE_ORDERING = "channels_last"


class DataLoaderError(Exception):
    pass

def get_pairs_from_paths(images_path, segs_path, ignore_non_matching=False):
    """ Find all the images from the images_path directory and
        the segmentation images from the segs_path directory
        while checking integrity of data """

    ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png"]
    ACCEPTABLE_SEGMENTATION_FORMATS = [".png"]

    image_files = []
    segmentation_files = {}

    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append((file_name, file_extension, os.path.join(images_path, dir_entry)))

    for dir_entry in os.listdir(segs_path):
        if os.path.isfile(os.path.join(segs_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_SEGMENTATION_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            if file_name in segmentation_files:
                raise DataLoaderError("Segmentation file with filename {0} already exists and is ambiguous to resolve with path {1}. Please remove or rename the latter.".format(file_name, os.path.join(segs_path, dir_entry)))
            segmentation_files[file_name] = (file_extension, os.path.join(segs_path, dir_entry))

    return_value = []
    # Match the images and segmentations
    for image_file, _, image_full_path in image_files:
        if image_file in segmentation_files:
            return_value.append((image_full_path, segmentation_files[image_file][1]))
        elif ignore_non_matching:
            continue
        else:
            # Error out
            raise DataLoaderError("No corresponding segmentation found for image {0}.".format(image_full_path))

    return return_value


def resize_padding(img):
    resize_ratio = 1.0
    old_size = img.shape[:2]
    desired_size = max(old_size)
    new_size = tuple([int(x * resize_ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return img


def resize_crop(img):
    resize_ratio = 1.0
    img_h, img_w, = img.shape[:2]
    h = w = max(img_h, img_w)
    input_ratio = img_w / img_h

    if input_ratio > resize_ratio:
        img_wr = int(input_ratio * h)
        img_hr = h
        img = cv2.resize(img, (img_wr, img_hr))
        x1 = int((img_wr - w) / 2)
        x2 = x1 + w
        img = img[:, x1:x2, :]
    if input_ratio < resize_ratio:
        img_wr = w
        img_hr = int(w / input_ratio)
        img = cv2.resize(img, (img_wr , img_hr))
        y1 = int((img_hr - h) / 2)
        y2 = y1 + h
        img = img[y1:y2, :, :]
    if input_ratio == resize_ratio:
        img = cv2.resize(img, (w, h))

    return img


def get_image_array(image_input, width, height, img_norm="sub_mean",
                  ordering="channels_first"):
    """ Load image array from input """

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif type(image_input) is str:
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_image_array: path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, 1)
    else:
        raise DataLoaderError("get_image_array: Can't process input type {0}".format(str(type(image_input))))

    if img_norm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif img_norm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        # img[:, :, 0] = (img[:, :, 0] - 87.195) / 70.900
        # img[:, :, 1] = (img[:, :, 1] - 86.408) / 68.489
        # img[:, :, 2] = (img[:, :, 2] - 89.585) / 70.921

        # img[:, :, 0] = (img[:, :, 0] * (0.27804186 ** 2) )
        # img[:, :, 1] = (img[:, :, 1] * (0.26858492 ** 2) )
        # img[:, :, 2] = (img[:, :, 2] * (0.27812227 ** 2) )

        cc = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
        img[:, :, 0] = cc
        img = img[:, :, 0]
        # img[:, :, 1] = cc
        # img[:, :, 2] = cc
        # img = img[:, :, ::-1]
    elif img_norm == "sub_mean_old":
        img = resize_padding(img)
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img[:, :, ::-1]
    elif img_norm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img/255.0
        # img = img[:, :, ::-1]
    elif img_norm == "sub_mean_new_resize":
        img = crop_and_resize(img, width, height)
        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img[:, :, ::-1]
    elif img_norm == "new_new":
        # img = cv2.resize(img, (width, height))
        # img = img.astype(np.float32)
        # img[:, :, 0] = (img[:, :, 0] - 121.29154027) / 68.12240857
        # img[:, :, 1] = (img[:, :, 1] - 117.34863752) / 67.22203485
        # img[:, :, 2] = (img[:, :, 2] - 111.89637408) / 69.61162893
        # img = img[:, :, ::-1]
        img = resize_padding(img)
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        cc = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
        img[:, :, 0] = cc
        img = img[:, :, 0]
    if ordering == "channels_first":
        img = np.rollaxis(img, 2, 0)

    return img


def get_segmentation_array(image_input, nClasses, width, height, no_reshape=False):
    """ Load segmentation array from input """

    seg_labels = np.zeros((height, width, nClasses))

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif type(image_input) is str:
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_segmentation_array: path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, 1)
    else:
        raise DataLoaderError("get_segmentation_array: Can't process input type {0}".format(str(type(image_input))))

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    img = img[:, :, 0]

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)

    if not no_reshape:
        seg_labels = np.reshape(seg_labels, (width*height, nClasses))

    return seg_labels


def verify_segmentation_dataset(images_path, segs_path, n_classes, show_all_errors=False):
    try:
        img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
        if not len(img_seg_pairs):
            print("Couldn't load any data from images_path: {0} and segmentations path: {1}".format(images_path, segs_path))
            return False

        return_value = True
        for im_fn, seg_fn in tqdm(img_seg_pairs):
            img = cv2.imread(im_fn)
            seg = cv2.imread(seg_fn)
            # Check dimensions match
            if not img.shape == seg.shape:
                return_value = False
                print("The size of image {0} and its segmentation {1} doesn't match (possibly the files are corrupt).".format(im_fn, seg_fn))
                if not show_all_errors:
                    break
            else:
                max_pixel_value = np.max(seg[:, :, 0])
                if max_pixel_value >= n_classes:
                    return_value = False
                    print("The pixel values of the segmentation image {0} violating range [0, {1}]. Found maximum pixel value {2}".format(seg_fn, str(n_classes - 1), max_pixel_value))
                    if not show_all_errors:
                        break
        if return_value:
            print("Dataset verified! ")
        else:
            print("Dataset not verified!")
        return return_value
    except DataLoaderError as e:
        print("Found error during data loading\n{0}".format(str(e)))
        return False

def image_segmentation_generator(images_path, segs_path, batch_size,
                                 n_classes, input_height, input_width,
                                 output_height, output_width,
                                 do_augment=False):

    img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
    random.shuffle(img_seg_pairs)
    zipped = itertools.cycle(img_seg_pairs)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = next(zipped)

            im = cv2.imread(im, 1)
            seg = cv2.imread(seg, 1)

            if do_augment:
                im, seg[:, :, 0] = augment_seg(im, seg[:, :, 0])

            X.append(get_image_array(im, input_width, input_height, img_norm=IMAGE_NORM, ordering=IMAGE_ORDERING))
            Y.append(get_segmentation_array(
                seg, n_classes, output_width, output_height))

        yield np.array(X), np.array(Y)
