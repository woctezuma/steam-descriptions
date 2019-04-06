# Objective: define a ground truth consisting of clusters of games which share genres or tags

import steamtags


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
    retrieval_ground_truth, _ = steamtags.load()

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
    _, retrieval_ground_truth = steamtags.load()

    retrieval_score = compute_retrieval_score_based_on_sharing_meta_data(query_app_ids,
                                                                         reference_app_id_counters,
                                                                         retrieval_ground_truth,
                                                                         meta_data_str='tags',
                                                                         num_elements_displayed=num_elements_displayed,
                                                                         verbose=verbose)

    return retrieval_score


if __name__ == '__main__':
    genres_dict, tags_dict = steamtags.load()
