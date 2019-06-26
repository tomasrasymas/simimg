from config import get_config
from src import helpers
import os
import numpy as np
from src.simimg_model import get_model

config = get_config()


class Img2Vec:
    def __init__(self, image_shape=config.IMAGE_SHAPE):
        self.image_shape = image_shape
        self.model, self.process_input = get_model(pooling='avg', input_shape=image_shape)

    def get_vector(self, image, preprocess=True):
        if isinstance(image, str):
            image = helpers.load_image_from_file(file_path=image,
                                                 image_shape=self.image_shape[:-1])

        image_x = np.copy(image)

        if preprocess:
            image_x = np.expand_dims(image_x, axis=0)
            image_x = self.process_input(image_x)

        output = self.model.predict(image_x)

        return output

    def get_vector_batch(self, image):
        img = list(image)
        for idx in range(len(img)):
            if isinstance(img[idx], str):
                img[idx] = helpers.load_image_from_file(file_path=img[idx],
                                                        image_shape=self.image_shape[:-1])

        img = np.array(img)

        output = self.get_vector(image=img, preprocess=False)
        return output

    def vectors_from_path(self, images_path):
        for subdir, dirs, files in os.walk(images_path):
            for file in files:
                image_path = os.path.join(subdir, file)
                if helpers.is_image_file(file_path=image_path):
                    yield image_path, self.get_vector(image=image_path)

    def vectors_from_path_batch(self, images_path, batch_size=1):
        all_files = helpers.get_path_image_files(images_path)
        files_batches = helpers.array_to_batches(all_files, batch_size)

        for batch in files_batches:
            vectors = self.get_vector_batch(batch)
            yield batch, vectors
