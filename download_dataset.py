import os
from config import get_config
import shutil
import requests
import zipfile

config = get_config()


def download_gdrive_file(file_id, destination):
    base_url = 'https://drive.google.com/uc?export=download'

    session = requests.Session()
    response = session.get(base_url, params={'id': file_id}, stream=True)
    token = None

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value

    if token:
        params = {'id': file_id,
                  'confirm': token}

        response = session.get(base_url, params=params, stream=True)

    with open(destination, 'wb') as f:
        shutil.copyfileobj(response.raw, f)

    return destination


def main():
    if os.path.isdir(config.DATASET_DEEPFASHION_PATH):
        shutil.rmtree(config.DATASET_DEEPFASHION_PATH)

    os.makedirs(config.DATASET_DEEPFASHION_PATH)

    for file_name, file_id in config.DATASET_FILES_TO_DOWNLOAD.items():
        print('Downloading %s...' % file_name)

        file_path = os.path.join(config.DATASET_DEEPFASHION_PATH, file_name)
        download_gdrive_file(file_id=file_id,
                             destination=file_path)

        if file_name.endswith('.zip'):
            print('Unzipping %s...' % file_name)

            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(config.DATASET_DEEPFASHION_PATH)

            os.remove(file_path)

    print('Done!')


if __name__ == '__main__':
    main()