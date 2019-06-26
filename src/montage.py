import imutils
from src import helpers
import math


def build_montages(images, image_shape, montage_size=None):
    if isinstance(images, str):
        images = helpers.load_images_from_path(images_path=images,
                                               image_shape=image_shape)
    elif isinstance(images, list):
        if isinstance(images[0], str):
            tmp = []
            for i in images:
                tmp.append(helpers.load_image_from_file(i, image_shape))
            images = tmp

    if not montage_size:
        m_size = math.ceil(math.sqrt(len(images)))
        montage_size = (m_size, m_size)

    montages = imutils.build_montages(images, image_shape, montage_size)

    return montages