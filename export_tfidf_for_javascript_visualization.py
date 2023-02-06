# Objective: match store descriptions with Term Frequency * Inverse Document Frequency (Tf-Idf).
# Reference: https://github.com/woctezuma/steam-descriptions/blob/master/sentence_baseline.py

import logging

import steamspypi
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import MatrixSimilarity

from benchmark_utils import get_app_name, load_benchmarked_app_ids, print_ranking
from doc2vec_model import reformat_similarity_scores_for_doc2vec
from sentence_models import print_most_similar_sentences
from utils import load_game_names, load_tokens


def load_input():
    game_names, _ = load_game_names(include_genres=False, include_categories=False)

    steam_tokens = load_tokens()

    app_ids = [int(app_id) for app_id in steam_tokens]

    return game_names, steam_tokens, app_ids


def train_tfidf_model(
    steam_tokens,
    no_below=5,
    no_above=0.5,
    normalize_vectors=False,
    num_best=10,
):
    documents = list(steam_tokens.values())

    dct = Dictionary(documents)
    dct.filter_extremes(no_below=no_below, no_above=no_above)

    print('Corpus as Bag-of-Words')
    corpus = [dct.doc2bow(doc) for doc in documents]

    print('Model: Term Frequency * Inverse Document Frequency (Tf-Idf)')
    model = TfidfModel(corpus, id2word=dct, normalize=normalize_vectors)

    index = MatrixSimilarity(model[corpus], num_best=num_best, num_features=len(dct))

    return dct, model, index


def match_queries(
    query_app_ids,
    game_names,
    steam_tokens,
    app_ids,
    dct,
    model,
    index,
    num_items_displayed=10,
    similarity_threshold=0.6,
    verbose=False,
):
    matches_as_app_ids = []

    for query_count, query_app_id in enumerate(query_app_ids):
        print(
            '[{}/{}] Query appID: {} ({})'.format(
                query_count + 1,
                len(query_app_ids),
                query_app_id,
                get_app_name(query_app_id, game_names),
            ),
        )

        try:
            query = steam_tokens[str(query_app_id)]
        except KeyError:
            print(f'Skipping query appID: {query_app_id}')
            similar_app_ids = []
            matches_as_app_ids.append(similar_app_ids)
            continue

        vec_bow = dct.doc2bow(query)

        vec_lsi = model[vec_bow]
        sims = index[vec_lsi]

        similarity_scores_as_tuples = [(str(app_ids[i]), sim) for (i, sim) in sims]
        if verbose:
            print(similarity_scores_as_tuples)

        # Filter out matches with a similarity score less than the user-chosen threshold
        similarity_scores_as_tuples = [
            (element, sim)
            for (element, sim) in similarity_scores_as_tuples
            if sim > similarity_threshold
        ]

        similarity_scores = reformat_similarity_scores_for_doc2vec(
            similarity_scores_as_tuples,
        )

        similar_app_ids = print_most_similar_sentences(
            similarity_scores,
            num_items_displayed=num_items_displayed,
            verbose=False,
        )
        matches_as_app_ids.append(similar_app_ids)

    return matches_as_app_ids


def export_for_javascript_visualization(
    query_app_ids,
    matches_as_app_ids,
    output_file_name='data/data.js',
    game_names=None,
):
    # TODO https://github.com/aesuli/word2vec_exploration/blob/master/doc_similarity_graph.py

    if game_names is None:
        game_names, _ = load_game_names(include_genres=False, include_categories=False)

    # Ensure appIDs are stored as integers (rather than strings)

    query_app_ids = [
        int(app_id)
        for (i, app_id) in enumerate(query_app_ids)
        if len(matches_as_app_ids[i]) > 1
    ]

    new_matches_as_app_ids = []
    for element in matches_as_app_ids:
        if len(element) > 1:
            new_element = [int(app_id) for app_id in element]
            new_matches_as_app_ids.append(new_element)
    matches_as_app_ids = new_matches_as_app_ids

    if not (len(query_app_ids) == len(matches_as_app_ids)):
        raise AssertionError()

    # Keep track of all the appIDs

    displayed_app_ids = set(query_app_ids)
    for element in matches_as_app_ids:
        displayed_app_ids.update(element)

    # Remove appIDs absent from our database

    displayed_app_ids = list(
        set(displayed_app_ids).intersection(int(app_id) for app_id in game_names),
    )

    # Keep track of all the game names

    displayed_game_names = [game_names[str(app_id)] for app_id in displayed_app_ids]

    # Use consecutive numbers to index appIDs

    nodetoidx = {}
    for i, app_id in enumerate(displayed_app_ids):
        nodetoidx[app_id] = i

    # Write to disk

    with open(output_file_name, 'w', encoding='utf8') as f:
        f.write('var nodes= [\n')

        for game_name in displayed_game_names:
            fixed_game_name = game_name.replace('"', '\'')
            current_str = '{"name":"' + fixed_game_name + '"},\n'
            f.write(current_str)

        f.write(']\nvar links = [\n')

        for i, query_app_id in enumerate(query_app_ids):
            source_ind = nodetoidx[query_app_id]
            for matched_app_id in matches_as_app_ids[i]:
                target_ind = nodetoidx[matched_app_id]
                if target_ind != source_ind:
                    current_str = (
                        '{ source: nodes['
                        + str(source_ind)
                        + '], target: nodes['
                        + str(target_ind)
                        + ']},\n'
                    )
                    f.write(current_str)

        f.write(']\n')

    return


def apply_workflow(
    query_app_ids=None,
    num_items_displayed=10,
    similarity_threshold=0.2,
):
    if query_app_ids is None:
        query_app_ids = load_benchmarked_app_ids(append_hard_coded_app_ids=True)

    query_app_ids = [int(app_id) for app_id in query_app_ids]

    game_names, _ = load_game_names(include_genres=False, include_categories=False)
    query_app_ids = list(
        set(query_app_ids).intersection(int(app_id) for app_id in game_names),
    )

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO,
    )

    game_names, steam_tokens, app_ids = load_input()

    dct, model, index = train_tfidf_model(steam_tokens, num_best=num_items_displayed)

    matches_as_app_ids = match_queries(
        query_app_ids,
        game_names,
        steam_tokens,
        app_ids,
        dct,
        model,
        index,
        num_items_displayed,
        similarity_threshold,
    )

    export_for_javascript_visualization(query_app_ids, matches_as_app_ids)

    print_ranking(query_app_ids, matches_as_app_ids, only_print_banners=True)

    return


def main(num_query_app_ids=100, num_items_displayed=10, similarity_threshold=0.2):
    # Data is already sorted by decreasing number of owners.
    data = steamspypi.load()
    all_app_ids_sorted_by_num_owners = [int(app_id) for app_id in data]

    query_app_ids = all_app_ids_sorted_by_num_owners[:num_query_app_ids]

    apply_workflow(
        query_app_ids,
        num_items_displayed=num_items_displayed,
        similarity_threshold=similarity_threshold,
    )

    return


if __name__ == '__main__':
    main()
