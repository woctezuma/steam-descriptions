import json
import logging

from benchmark_utils import get_app_name
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
        with open(get_unique_games_file_name(), 'r') as f:
            sim_dict = json.load(f)
    except FileNotFoundError:
        sim_dict = dict()

    for query_count, query_app_id in enumerate(query_app_ids):

        if query_app_id in sim_dict.keys():
            continue

        print('[{}/{}] Query appID: {} ({})'.format(query_count + 1,
                                                    len(query_app_ids),
                                                    query_app_id,
                                                    get_app_name(query_app_id, game_names)))

        query = steam_tokens[str(query_app_id)]

        if len(query) == 0:
            continue

        vec_bow = dct.doc2bow(query)
        sims = index[model[vec_bow]]

        similarity_scores_as_tuples = [(int(app_ids[i]), sim) for (i, sim) in sims]

        second_best_similarity_score_as_tuple = similarity_scores_as_tuples[1]

        second_best_matched_app_id = int(second_best_similarity_score_as_tuple[0])

        second_best_similarity_score = second_best_similarity_score_as_tuple[1]

        sim_dict[int(query_app_id)] = dict()
        sim_dict[int(query_app_id)]['app_id'] = second_best_matched_app_id
        sim_dict[int(query_app_id)]['similarity'] = second_best_similarity_score

        save_to_disk = bool((query_count + 1) % 100 == 0)

        if save_to_disk:
            with open(get_unique_games_file_name(), 'w') as f:
                json.dump(sim_dict, f)

    return sim_dict


def main(num_items_displayed=2,
         similarity_threshold=0.2):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    game_names, steam_tokens, app_ids = load_input()

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

    # TODO use similarity_threshold to display a small part of the results

    return


if __name__ == '__main__':
    main()
