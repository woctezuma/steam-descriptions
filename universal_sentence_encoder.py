from time import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

from benchmark_utils import get_top_100_app_ids, print_ranking
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


def prepare_knn_search(label_database=None, use_cosine_similarity=True):
    if label_database is None:
        label_database = load_embedded_descriptions()

    if use_cosine_similarity:
        knn = NearestNeighbors(metric='cosine', algorithm='brute')
        knn.fit(label_database)
    else:
        knn = NearestNeighbors(algorithm='brute')
        knn.fit(label_database)

    return knn


def get_query_descriptor(query_app_id, label_database=None, app_ids=None):
    if label_database is None:
        label_database = load_embedded_descriptions()

    if app_ids is None:
        app_ids = load_embedding_app_ids()

    try:
        query_des = label_database[[app_ids.index(query) for query in query_app_id]]
    except TypeError:
        query_des = label_database[app_ids.index(query_app_id)]

    return query_des


def perform_knn_search_with_vectors_as_input(query_des, knn, num_neighbors=10):
    start = time()

    if len(query_des.shape) == 1:
        # Sci-Kit Learn with cosine similarity. Reshape data as it contains a single sample.
        _, matches = knn.kneighbors(query_des.reshape(1, -1), n_neighbors=num_neighbors)
    else:
        _, matches = knn.kneighbors(query_des, n_neighbors=num_neighbors)

    print('Elapsed time: {:.2f} s'.format(time() - start))

    return matches


def transform_matches_to_app_ids(matches, app_ids=None):
    if app_ids is None:
        app_ids = load_embedding_app_ids()

    matches_as_app_ids = []

    for row in matches:
        reference_app_id_counter = [app_ids[element] for element in row]
        matches_as_app_ids.append(reference_app_id_counter)

    return matches_as_app_ids


def print_formatted_knn_search_results(formatted_results, query_app_id=None):
    for counter, ranking in enumerate(formatted_results):

        if query_app_id is not None:
            print('\nQuery: {}'.format(query_app_id[counter]))
        else:
            print('\nQuery: not available')

        for rank, app_info in enumerate(ranking):
            app_id = app_info[0]
            app_name = app_info[1]
            print('{:2}) {} ({})'.format(rank + 1, app_name, app_id))

    return


def perform_knn_search_with_app_ids_as_input(query_app_ids,
                                             label_database=None,
                                             app_ids=None,
                                             knn=None,
                                             use_cosine_similarity=True,  # only taken into account if 'knn' is None
                                             num_neighbors=10):
    if label_database is None:
        label_database = load_embedded_descriptions()

    if app_ids is None:
        app_ids = load_embedding_app_ids()

    if knn is None:
        knn = prepare_knn_search(label_database, use_cosine_similarity=use_cosine_similarity)

    # From query appID to query feature vector
    query_des = get_query_descriptor(query_app_ids, label_database, app_ids)

    # Matching of feature vectors
    matches = perform_knn_search_with_vectors_as_input(query_des, knn, num_neighbors)

    # From feature matches to appID matches
    matches_as_app_ids = transform_matches_to_app_ids(matches, app_ids)

    return matches_as_app_ids


if __name__ == '__main__':
    top_100_app_ids = get_top_100_app_ids()

    query_app_ids = top_100_app_ids[:3]
    use_cosine_similarity = True
    num_neighbors = 10
    only_print_banners = True

    matches_as_app_ids = perform_knn_search_with_app_ids_as_input(query_app_ids,
                                                                  use_cosine_similarity=use_cosine_similarity,
                                                                  num_neighbors=num_neighbors)

    print_ranking(query_app_ids,
                  matches_as_app_ids,
                  num_elements_displayed=num_neighbors,
                  only_print_banners=only_print_banners)
