import json
import logging

from benchmark_utils import get_app_name, get_banner_url, get_store_url
from export_tfidf_for_javascript_visualization import load_input, train_tfidf_model


def get_unique_games_file_name():
    unique_games_file_name = 'data/unique_games.json'

    return unique_games_file_name


def populate_database(query_app_ids,
                      game_names,
                      steam_tokens,
                      app_ids,
                      dct,
                      model,
                      index):
    try:
        sim_dict = load_sim_dict()
    except FileNotFoundError:
        sim_dict = dict()

    query_app_ids = set(query_app_ids).difference([int(app_id) for app_id in sim_dict.keys()])
    query_app_ids = sorted(list(query_app_ids), key=int)

    for query_count, query_app_id in enumerate(query_app_ids):

        if str(query_app_id) in sim_dict.keys():
            continue

        print('[{}/{}] Query appID: {} ({})'.format(query_count + 1,
                                                    len(query_app_ids),
                                                    query_app_id,
                                                    get_app_name(query_app_id, game_names)))

        query = steam_tokens[str(query_app_id)]

        # Typically for empty descriptions, e.g. with appID: 3300 (Bejeweled 2 Deluxe)
        if len(query) == 0:
            print('No description input for appID = {}'.format(query_app_id))
            continue

        vec_bow = dct.doc2bow(query)

        # Typically for descriptions in Chinese, e.g. with appID: 859200 (破东荒 - Chaos Of East)
        if len(vec_bow) == 0:
            print('No Bag-of-Words input for appID = {}'.format(query_app_id))
            continue

        sims = index[model[vec_bow]]

        similarity_scores_as_tuples = [(int(app_ids[i]), sim) for (i, sim) in sims]

        second_best_similarity_score_as_tuple = similarity_scores_as_tuples[1]

        second_best_matched_app_id = int(second_best_similarity_score_as_tuple[0])

        second_best_similarity_score = second_best_similarity_score_as_tuple[1]

        sim_dict[query_app_id] = dict()
        sim_dict[query_app_id]['app_id'] = second_best_matched_app_id
        sim_dict[query_app_id]['similarity'] = second_best_similarity_score

        save_to_disk = bool((query_count + 1) % 300 == 0)

        if save_to_disk:
            with open(get_unique_games_file_name(), 'w') as f:
                json.dump(sim_dict, f)

    with open(get_unique_games_file_name(), 'w') as f:
        json.dump(sim_dict, f)

    return sim_dict


def load_sim_dict():
    with open(get_unique_games_file_name(), 'r') as f:
        sim_dict = json.load(f)

    return sim_dict


def print_unique_games(sim_dict,
                       similarity_threshold,
                       game_names,
                       only_print_banners=False):
    # Markdown
    # Reference: https://stackoverflow.com/a/14747656
    image_link_str = '[<img alt="{}" src="{}" width="{}">]({})'
    image_width = 150

    sorted_app_ids = sorted(sim_dict.keys(), key=lambda x: sim_dict[x]['similarity'])

    unique_app_ids = []

    for counter, app_id in enumerate(sorted_app_ids):
        similarity_value = sim_dict[app_id]['similarity']
        if similarity_value <= similarity_threshold:
            unique_app_ids.append(app_id)

            app_name = get_app_name(app_id, game_names=game_names)
            if only_print_banners:
                # Markdown
                print(image_link_str.format(app_name, get_banner_url(app_id), image_width, get_store_url(app_id)))
            else:
                print('{}) similarity = {:.2f} ; appID = {} ({})'.format(counter + 1,
                                                                         similarity_value,
                                                                         app_id,
                                                                         app_name))

    return unique_app_ids


def main(num_items_displayed=2,
         similarity_threshold=0.2,
         update_sim_dict=False):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    game_names, steam_tokens, app_ids = load_input()

    if update_sim_dict:
        query_app_ids = list(set(app_ids).intersection(int(app_id) for app_id in game_names.keys()))
        query_app_ids = sorted(query_app_ids, key=int)

        dct, model, index = train_tfidf_model(steam_tokens,
                                              num_best=num_items_displayed)

        sim_dict = populate_database(query_app_ids,
                                     game_names,
                                     steam_tokens,
                                     app_ids,
                                     dct,
                                     model,
                                     index)
    else:
        sim_dict = load_sim_dict()

    unique_app_ids = print_unique_games(sim_dict, similarity_threshold, game_names)

    return


if __name__ == '__main__':
    main()
