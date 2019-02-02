# Objective: learn a Word2Vec model, then build a sentence embedding based on Word Mover Distance.

import operator

from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity

from sentence_models import print_most_similar_sentences
from utils import get_word_model_file_name, load_tokens


def reformat_similarity_scores_for_wmd(similarity_scores_as_tuples, candidates):
    similarity_scores = dict()
    for (app_id, my_tuple) in zip(candidates, sorted(similarity_scores_as_tuples, key=operator.itemgetter(0))):
        similarity_scores[app_id] = my_tuple[1]

    return similarity_scores


def compute_similarity_with_candidate_sentences_using_wmd(query_app_id, steam_tokens=None, model=None,
                                                          candidates=None):
    if steam_tokens is None:
        steam_tokens = load_tokens()

    if model is None:
        model = Word2Vec.load(get_word_model_file_name())

    constrain_search = (candidates is not None)

    query = steam_tokens[query_app_id]

    if constrain_search:
        documents = list(steam_tokens[i] for i in candidates)
    else:
        # Caveat: the Word Mover algorithm is painfully slow! Please consider constraining the search to few candidates!
        documents = list(steam_tokens.values())

    instance = WmdSimilarity(documents, model.wv, num_best=10)

    similarity_scores_as_tuples = instance[query]

    similarity_scores = reformat_similarity_scores_for_wmd(similarity_scores_as_tuples, candidates)
    print_most_similar_sentences(similarity_scores)

    return similarity_scores


if __name__ == '__main__':
    query_app_id = '583950'  # Artifact

    # AppIDs found with compute_similarity_with_all_other_steam_sentences() in sentence_models.py
    candidates = ['562540', '687280', '878940', '485000', '381560',
                  '336040', '867420', '851040', '491670', '710700']

    compute_similarity_with_candidate_sentences_using_wmd(query_app_id, candidates=candidates)
