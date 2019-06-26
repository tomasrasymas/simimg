import argparse
import os
from src.img_2_vec import Img2Vec
from src import helpers

img2vec = Img2Vec()


def main(to_process, output):
    output_data = []
    if os.path.isfile(to_process):
        vec = img2vec.get_vector(image=to_process)
        print(vec[0].shape)
        print(vec[0])
    elif os.path.isdir(to_process):
        for v_file, v_vector in img2vec.vectors_from_path(images_path=to_process):
            print('%s - %s' % (v_file, v_vector))

            output_data.append({
                'file_name': v_file,
                'vector': v_vector[0]
            })

        helpers.write_to_json(file_path=output, data=output_data)
    else:
        raise FileNotFoundError('File %s not found.' % to_process)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Images to vectors',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--process', dest='to_process', type=str, metavar='',
                        required=False, help='Path of file/directory to process', default='datasets/deepfashion/img/36_Plaid_Shirt_Dress/img_00000001.jpg')
    parser.add_argument('-o', '--output', dest='output', type=str, metavar='',
                        required=False, help='Path of output file', default='output.json')

    args = parser.parse_args()

    print('*' * 50)
    for i in vars(args):
        print(str(i) + ' - ' + str(getattr(args, i)))

    print('*' * 50)

    main(to_process=args.to_process, output=args.output)