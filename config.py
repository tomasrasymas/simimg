import os


class Config:
    DATA_PATH = 'data'
    DATASET_DEEPFASHION_PATH = os.path.join(DATA_PATH, 'deepfashion')
    DATASET_FILES_TO_DOWNLOAD = {
        'list_attr_cloth': '0B7EVK8r0v71pYnBKQVBOaHR1WWs',
        'list_attr_img': '0B7EVK8r0v71pWXE4QWotX2hxQ1U',
        'list_category_cloth': '0B7EVK8r0v71pWnFiNlNGTVloLUk',
        'list_category_img': '0B7EVK8r0v71pTGNoWkhZeVpzbFk',
        'img.zip': '0B7EVK8r0v71pa2EyNEJ0dE9zbU0',
        'README': '0B7EVK8r0v71pdERsaTdrbS1VbzA'
    }
    FEATURES_PATH = os.path.join(DATA_PATH, 'features')
    FEATURES_ATTRIBUTES_PATH = os.path.join(FEATURES_PATH, 'attributes.csv')
    FEATURES_CATEGORIES_PATH = os.path.join(FEATURES_PATH, 'categories.csv')
    FEATURES_TYPE_PATH = os.path.join(FEATURES_PATH, 'type.csv')
    # IMAGE_SIZE = (244, 244, 3) # resnet50
    # IMAGE_SIZE = (224, 224, 3) # nasnet
    IMAGE_SHAPE = (75, 75, 3) # deep fashion
    IMAGE_EXTENSIONS = ['.jpg', 'jpeg', '.png']
    TB_PROJECTOR_CONFIG_TEMPLATE = '''
    embeddings {
      tensor_path: "%s"
      metadata_path: "%s"
      sprite {
        image_path: "%s"
        single_image_dim: %s
        single_image_dim: %s
      }
    }'''
    TB_PROJECTOR_MONTAGE_FILE_NAME = 'montage.png'
    TB_PROJECTOR_FEATURES_FILE_NAME = 'features.tsv'
    TB_PROJECTOR_METADATA_FILE_NAME = 'metadata.tsv'
    TB_PROJECTOR_PROJECTOR_CONFIG_FILE_NAME = 'projector_config.pbtxt'
    PICKLE_DATA_FILE_NAME = os.path.join(DATA_PATH, 'features_db.pckl')
    TRAINED_MODEL_PATH = os.path.join(DATA_PATH, 'best_model.h5')
    TRAINING_BATCH_SIZE = 64
    MODEL_OUTPUT_LAYERS = ['dense_2']


class DevelopmentConfig(Config):
    pass


class TestConfig(Config):
    pass


class ProductionConfig(Config):
    pass


def get_config():
    env = os.environ.get('env', None)

    if env == 'development':
        return DevelopmentConfig
    elif env == 'production':
        return ProductionConfig
    elif env == 'test':
        return TestConfig
    else:
        return DevelopmentConfig
