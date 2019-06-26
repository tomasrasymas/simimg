import argparse
from src.img_2_vec import Img2Vec
import numpy as np
from src import helpers
import config

config = config.get_config()
img_2_vec = Img2Vec()


def generate(files_path, batch_size=100):
    tmp_data = []
    for idx, (img_path, img_vec) in enumerate(img_2_vec.vectors_from_path_batch(files_path,
                                                                                batch_size=batch_size)):
        print('%s batch loaded' % (idx + 1))
        for i in range(len(img_path)):
            tmp_data.append([img_path[i], img_vec[i]])

    data = np.array(tmp_data)

    helpers.object_to_pickle_file(config.PICKLE_DATA_FILE_NAME, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate features DB',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', dest='files_path', type=str, metavar='',
                        required=False, help='Path of images files to put tov DB', default='data/montage/')
    parser.add_argument('-b', '--batch', dest='batch_size', type=int, required=False, metavar='',
                        help='Batch size', default=5)

    args = parser.parse_args()

    print('*' * 50)
    for i in vars(args):
        print(str(i) + ' - ' + str(getattr(args, i)))

    print('*' * 50)

    generate(args.files_path, args.batch_size)

    print('Done')