from tensorflow.python.keras.preprocessing import image
from config import get_config
import os
import json
import numpy as np
import csv
import cv2
import math
import shutil
import pickle
from scipy import spatial

config = get_config()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def load_image_from_file(file_path, image_shape=config.IMAGE_SHAPE):
    img = image.load_img(file_path, target_size=image_shape)
    img = image.img_to_array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def load_images_from_path(images_path, image_shape=config.IMAGE_SHAPE, with_path=False):
    images = []
    images_paths = []

    for subdir, dirs, files in os.walk(images_path):
        for file in files:
            if is_image_file(file_path=file):
                image_path = os.path.join(subdir, file)
                images_paths.append(image_path)
                images.append(load_image_from_file(image_path, image_shape=image_shape))

    if with_path:
        return images_paths, images

    return images


def get_path_image_files(path):
    image_files = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if is_image_file(file_path=file):
                image_path = os.path.join(subdir, file)
                image_files.append(image_path)

    return np.asarray(image_files)


def get_part_of_array(array, num_of_elements, shuffle=True):
    if shuffle:
        np.random.shuffle(array)

    return array[0:num_of_elements]


def is_image_file(file_path):
    for _ in config.IMAGE_EXTENSIONS:
        if file_path.endswith(_):
            return True

    return False


def write_to_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder)


def list_to_csv_file(file_name, data):
    with open(file_name, 'w') as fw:
        csv_writer = csv.writer(fw, delimiter='\t')

        for d in data:
            if isinstance(d, str):
                d = [[d]]

            csv_writer.writerows(d)


def text_to_text_file(file_name, text):
    with open(file_name, 'w') as f:
        f.write(text)


def delete_path_content(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


def array_to_batches(array, batch_size):
    num_batches = math.ceil((len(array) / batch_size))
    return np.array_split(array, num_batches)


def object_to_pickle_file(file_path, object):
    with open(file_path, 'wb') as f:
        pickle.dump(object, f)


def load_object_from_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def cosine_similarity(vec1, vec2):
    return 1 - spatial.distance.cosine(vec1, vec2)


def minkowski_similarity(vec1, vec2):
    return spatial.distance.minkowski(vec1, vec2)


def euclidean_similarity(vec1, vec2):
    return spatial.distance.euclidean(vec1, vec2)


def create_path_if_not_exists(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)


def one_hot_list(length, idx):
    one_hot = [0 for _ in range(length)]
    one_hot[idx] = 1
    return one_hot