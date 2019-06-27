from src.img_2_vec import Img2Vec
from config import get_config
import os
from src import helpers
import numpy as np
from src import montage
from src import display
import argparse
import datetime

config = get_config()


class SimilarSearch:
    def __init__(self, distance):
        self.img_2_vec = Img2Vec()
        self.data = None
        self.distance = distance

    def load(self):
        print('Loading %s' % datetime.datetime.now())

        if os.path.isfile(config.PICKLE_DATA_FILE_NAME):
            self.data = helpers.load_object_from_pickle_file(config.PICKLE_DATA_FILE_NAME)

        print('Loading done %s' % datetime.datetime.now())

    def find(self, image, num_of_results=2):
        print('Searching %s' % datetime.datetime.now())

        vec = self.img_2_vec.get_vector(image=image)

        if self.distance == 'cosine':
            distances = np.array([[helpers.cosine_similarity(vec, d[1])] for d in self.data])
        elif self.distance == 'euclidean':
            distances = np.array([[helpers.euclidean_similarity(vec, d[1])] for d in self.data])
        elif self.distance == 'minkowski':
            distances = np.array([[helpers.minkowski_similarity(vec, d[1])] for d in self.data])
        else:
            distances = np.array([[helpers.cosine_similarity(vec, d[1])] for d in self.data])

        data_with_distances = np.hstack((self.data, distances))
        data_with_distances = data_with_distances[data_with_distances[:, 2].argsort()]

        print('Searching done %s' % datetime.datetime.now())

        return data_with_distances[-num_of_results - 1:-1, :].tolist()

    def find_and_display(self, sample_file_path, num_of_results=2, images_size=(100, 100)):
        found_files = self.find(sample_file_path, num_of_results)

        found_files = [[sample_file_path, [], 0]] + found_files
        img_montage = montage.build_montages([i[0] for i in found_files],
                                             images_size)
        display.montages(img_montage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search similar images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--num', dest='similar_num', type=int, required=False, metavar='',
                        help='Number of similar images to find', default=10)
    parser.add_argument('-d', '--distance', dest='distance', type=str, required=False, metavar='',
                        help='Distance algorithm: cosine | euclidean | minkowski', default='cosine')

    args = parser.parse_args()

    print('*' * 50)
    for i in vars(args):
        print(str(i) + ' - ' + str(getattr(args, i)))

    print('*' * 50)

    search = SimilarSearch(args.distance)
    search.load()

    while True:
        sample_file_path = input('Sample file path >>> ')
        while not sample_file_path:
            print('Prompt should not be empty!')
            sample_file_path = input('Sample file path >>> ')

        print('=' * 40)

        search.find_and_display(sample_file_path, num_of_results=args.similar_num, images_size=(100, 100))