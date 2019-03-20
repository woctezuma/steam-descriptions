import numpy as np

from utils import get_data_folder


def get_embedding_file_name_prefix():
    embedding_file_name_prefix = 'universal-sentence-encoder-'
    return embedding_file_name_prefix


def get_embedded_description_file_name():
    embedded_description_file_name = get_data_folder() + get_embedding_file_name_prefix() + 'features.npy'
    return embedded_description_file_name


def get_embedding_app_id_file_name():
    embedding_app_id_file_name = get_data_folder() + get_embedding_file_name_prefix() + 'appids.txt'
    return embedding_app_id_file_name


def load_embedding_app_ids():
    with open(get_embedding_app_id_file_name(), 'r', encoding='utf-8') as f:
        app_id_list_as_str = f.readlines()[0].strip()

    app_id_list = [int(app_id.strip('\'')) for app_id in app_id_list_as_str.strip('[]').split(', ')]

    return app_id_list


def load_embedded_descriptions():
    message_embeddings = np.load(get_embedded_description_file_name())
    return message_embeddings


if __name__ == '__main__':
    embedding_app_ids = load_embedding_app_ids()
    embedded_descriptions = load_embedded_descriptions()

    print('Done.')
