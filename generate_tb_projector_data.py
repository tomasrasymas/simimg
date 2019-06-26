from src import montage
from src import display
from src import helpers
from src.img_2_vec import Img2Vec
from config import get_config
from shutil import copyfile
import argparse
import cv2
import os

config = get_config()
img2vec = Img2Vec()

SEPARATOR = '-|-'


def main(files_path, output_files_path, num_dataset, image_shape=(50, 50), only_copy=False):
    montage_file_path = os.path.join(output_files_path, config.TB_PROJECTOR_MONTAGE_FILE_NAME)
    features_file_path = os.path.join(output_files_path, config.TB_PROJECTOR_FEATURES_FILE_NAME)
    metadata_file_path = os.path.join(output_files_path, config.TB_PROJECTOR_METADATA_FILE_NAME)
    projector_config_file_path = os.path.join(output_files_path, config.TB_PROJECTOR_PROJECTOR_CONFIG_FILE_NAME)

    if num_dataset:
        helpers.delete_path_content(files_path)
        image_files = helpers.get_path_image_files(config.DATASET_DEEPFASHION_PATH)
        files_to_copy = helpers.get_part_of_array(array=image_files, num_of_elements=num_dataset)
        for file_path in files_to_copy:
            rename_to = os.path.join(files_path, os.path.basename(os.path.split(file_path)[0]) + SEPARATOR + os.path.split(file_path)[1])
            copyfile(file_path, rename_to)

    if only_copy:
        return

    images_paths, images = helpers.load_images_from_path(files_path, image_shape=config.IMAGE_SHAPE, with_path=True)

    helpers.list_to_csv_file(file_name=metadata_file_path,
                             data=[os.path.split(i)[1].split(SEPARATOR)[0] for i in images_paths])

    features = []

    for image in images:
        features.append(img2vec.get_vector(image=image))

    helpers.list_to_csv_file(file_name=features_file_path,
                             data=features)

    montages = montage.build_montages(images=images, image_shape=image_shape)
    cv2.imwrite(montage_file_path, montages[0])

    projector_config = config.TB_PROJECTOR_CONFIG_TEMPLATE % (config.TB_PROJECTOR_FEATURES_FILE_NAME,
                                                              config.TB_PROJECTOR_METADATA_FILE_NAME,
                                                              config.TB_PROJECTOR_MONTAGE_FILE_NAME,
                                                              image_shape[0],
                                                              image_shape[1])
    helpers.text_to_text_file(projector_config_file_path, projector_config)

    display.montages(montages=montages, scale=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Images to vectors',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', dest='files_path', type=str, metavar='',
                        required=False, help='Path of image files', default='data/montage/')
    parser.add_argument('-o', '--output', dest='output_files_path', type=str, metavar='',
                        required=False, help='Path of output files', default='data/projector/')
    parser.add_argument('-dn', '--dataset_num', dest='num_dataset', type=int, required=False, metavar='',
                        help='Number of random images to copy from dataset', default=10)
    parser.add_argument('-c', '--only_copy', dest='only_copy', action='store_true',
                        help='Only copy files and do not generate files for TB projector')

    args = parser.parse_args()

    print('*' * 50)
    for i in vars(args):
        print(str(i) + ' - ' + str(getattr(args, i)))

    print('*' * 50)

    main(args.files_path, args.output_files_path, args.num_dataset, only_copy=args.only_copy)