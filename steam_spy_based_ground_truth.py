# Objective: define a ground truth consisting of clusters of games which share genres or tags

import json
import time

import steamspypi


def get_data_folder():
    return 'data/'


def get_file_name_for_list_of_genres():
    return get_data_folder() + 'genre_keys.json'


def get_file_name_for_clustering_of_app_ids_by_genre():
    return get_data_folder() + 'genre_values.json'


def get_file_name_for_list_of_tags():
    return get_data_folder() + 'tag_keys.json'


def get_file_name_for_clustering_of_app_ids_by_tag():
    return get_data_folder() + 'tag_values.json'


def download_genre_and_tag_keys():
    # Genres and tags downloaded from SteamSpy API

    print('Downloading lists of genres and tags.')

    # Retrieve top 100 games played in the past 2 weeks

    data_request = dict()
    data_request['request'] = 'top100in2weeks'

    data = steamspypi.download(data_request)

    # Aggregate genres and tags

    genres = set()
    tags = set()

    for counter, app_id in enumerate(data):

        data_request = dict()
        data_request['request'] = 'appdetails'
        data_request['appid'] = app_id

        data_app_id = steamspypi.download(data_request)

        current_genres = set(s.strip() for s in data_app_id['genre'].split(','))
        current_tags = set(data_app_id['tags'].keys())

        genres.update(current_genres)
        tags.update(current_tags)

        # Allowed poll rate - 4 requests per second.
        # Reference: https://steamspy.com/api.php
        if counter % 4 == 0:
            print('{}/{}'.format(counter, len(data)))
            time.sleep(1)

    genres = sorted(genres)
    tags = sorted(tags)

    with open(get_file_name_for_list_of_genres(), 'w', encoding='utf8') as f:
        print('\n'.join(genres), file=f)

    with open(get_file_name_for_list_of_tags(), 'w', encoding='utf8') as f:
        print('\n'.join(tags), file=f)

    return genres, tags


def load_genre_keys():
    try:
        with open(get_file_name_for_list_of_genres(), 'r', encoding='utf8') as f:
            genres = [l.strip() for l in f.readlines()]

    except FileNotFoundError:
        genres, _ = download_genre_and_tag_keys()

    return genres


def load_tag_keys():
    try:
        with open(get_file_name_for_list_of_tags(), 'r', encoding='utf8') as f:
            tags = [l.strip() for l in f.readlines()]
    except FileNotFoundError:
        _, tags = download_genre_and_tag_keys()

    return tags


def populate_genres(genres):
    print('Downloading clusters of appIDs for each genre.')

    genres_dict = dict()

    for counter, current_genre in enumerate(genres):

        data_request = dict()
        data_request['request'] = 'genre'
        data_request['genre'] = current_genre

        data = steamspypi.download(data_request)

        genres_dict[current_genre] = list(int(app_id) for app_id in data.keys())

        # Allowed poll rate - 4 requests per second.
        # Reference: https://steamspy.com/api.php
        if counter % 4 == 0:
            print('{}/{}'.format(counter, len(genres)))
            time.sleep(1)

    with open(get_file_name_for_clustering_of_app_ids_by_genre(), 'w', encoding='utf8') as f:
        json.dump(genres_dict, f)

    return genres_dict


def populate_tags(tags):
    print('Downloading clusters of appIDs for each tag.')

    tags_dict = dict()

    for counter, current_tag in enumerate(tags):

        data_request = dict()
        data_request['request'] = 'tag'
        data_request['tag'] = current_tag

        data = steamspypi.download(data_request)

        tags_dict[current_tag] = list(int(app_id) for app_id in data.keys())

        # Allowed poll rate - 4 requests per second.
        # Reference: https://steamspy.com/api.php
        if counter % 4 == 0:
            print('{}/{}'.format(counter, len(tags)))
            time.sleep(1)

    with open(get_file_name_for_clustering_of_app_ids_by_tag(), 'w', encoding='utf8') as f:
        json.dump(tags_dict, f)

    return tags_dict


def load_genre_values():
    try:
        with open(get_file_name_for_clustering_of_app_ids_by_genre(), 'r', encoding='utf8') as f:
            genres_dict = json.load(f)
    except FileNotFoundError:
        genres = load_genre_keys()
        genres_dict = populate_genres(genres)

    return genres_dict


def load_tag_values():
    try:
        with open(get_file_name_for_clustering_of_app_ids_by_tag(), 'r', encoding='utf8') as f:
            tags_dict = json.load(f)
    except FileNotFoundError:
        tags = load_tag_keys()
        tags_dict = populate_tags(tags)

    return tags_dict


def compute_retrieval_score_based_on_sharing_meta_data(query_app_ids,
                                                       reference_app_id_counters,
                                                       retrieval_ground_truth,
                                                       meta_data_str='meta-data',
                                                       num_elements_displayed=10,
                                                       verbose=True):
    print('\nComputing retrieval score based on sharing {}.'.format(meta_data_str))

    retrieval_score = 0

    for query_counter, query_app_id in enumerate(query_app_ids):
        reference_app_id_counter = reference_app_id_counters[query_counter]

        meta_data_of_query = set(meta_data for meta_data in retrieval_ground_truth.keys()
                                 if int(query_app_id) in retrieval_ground_truth[meta_data])

        if len(meta_data_of_query) == 0:
            continue

        current_retrieval_score = 0
        for rank, app_id in enumerate(reference_app_id_counter):

            meta_data_of_reference = set(meta_data for meta_data in retrieval_ground_truth.keys()
                                         if int(app_id) in retrieval_ground_truth[meta_data])

            numerator = len(meta_data_of_query.intersection(meta_data_of_reference))
            denominator = len(meta_data_of_query.union(meta_data_of_reference))

            if app_id != query_app_id:
                current_retrieval_score += (numerator / denominator)

            if rank >= (num_elements_displayed - 1):
                retrieval_score += current_retrieval_score
                if verbose:
                    print('[appID={}] retrieval score based on sharing meta-data = {}'.format(query_app_id,
                                                                                              current_retrieval_score))
                break

    print('Total retrieval score based on sharing {} = {}'.format(meta_data_str, retrieval_score))

    return retrieval_score


def compute_retrieval_score_based_on_sharing_genres(query_app_ids,
                                                    reference_app_id_counters,
                                                    num_elements_displayed=10,
                                                    verbose=True):
    retrieval_ground_truth = load_genre_values()

    retrieval_score = compute_retrieval_score_based_on_sharing_meta_data(query_app_ids,
                                                                         reference_app_id_counters,
                                                                         retrieval_ground_truth,
                                                                         meta_data_str='genres',
                                                                         num_elements_displayed=num_elements_displayed,
                                                                         verbose=verbose)

    return retrieval_score


def compute_retrieval_score_based_on_sharing_tags(query_app_ids,
                                                  reference_app_id_counters,
                                                  num_elements_displayed=10,
                                                  verbose=True):
    retrieval_ground_truth = load_tag_values()

    retrieval_score = compute_retrieval_score_based_on_sharing_meta_data(query_app_ids,
                                                                         reference_app_id_counters,
                                                                         retrieval_ground_truth,
                                                                         meta_data_str='tags',
                                                                         num_elements_displayed=num_elements_displayed,
                                                                         verbose=verbose)

    return retrieval_score


if __name__ == '__main__':
    genres = load_genre_keys()
    tags = load_tag_keys()

    genres_dict = load_genre_values()
    tags_dict = load_tag_values()
