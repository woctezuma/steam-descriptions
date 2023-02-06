# Objective: learn a Doc2Vec model

import logging
import multiprocessing
import random
from time import time

import numpy as np
from gensim.models import doc2vec

from benchmark_utils import load_benchmarked_app_ids, print_ranking
from sentence_models import print_most_similar_sentences
from universal_sentence_encoder import (
    perform_knn_search_with_vectors_as_input,
    prepare_knn_search,
    transform_matches_to_app_ids,
)
from utils import get_doc_model_file_name, load_game_names, load_tokens
from word_model import compute_similarity_using_word2vec_model


def get_tag_prefix():
    return 'appID_'


def read_corpus(steam_tokens, game_tags=None, include_app_ids=True):
    for app_id, tokens in steam_tokens.items():
        doc_tag = []

        if include_app_ids:
            doc_tag += [get_tag_prefix() + str(app_id)]

        try:
            # Reference: https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e
            doc_tag += game_tags[app_id]
        except KeyError:
            print(f'AppID = {app_id} cannot be found in tag dictionary.')
        except TypeError:
            pass

        yield doc2vec.TaggedDocument(tokens, doc_tag)


def reformat_similarity_scores_for_doc2vec(
    similarity_scores_as_tuples,
    game_names=None,
):
    if game_names is None:
        game_names, _ = load_game_names()

    dummy_app_ids = []

    similarity_scores = {}
    for app_id, similarity_value in similarity_scores_as_tuples:
        if app_id.startswith(get_tag_prefix()):
            app_id = app_id[len(get_tag_prefix()) :]

        similarity_scores[str(app_id)] = similarity_value

        if str(app_id) not in game_names:
            dummy_app_ids.append(app_id)

    if len(dummy_app_ids) > 0:
        print(f'Dummy appIDs: {dummy_app_ids}')

    return similarity_scores


def train_doc_model_on_steam_tokens(model=None, steam_tokens=None, num_epochs=10):
    # You do not want to perform training this way, because training already happened when initializating the model
    # with Doc2Vec(documents). Moreover, calling train() several times messes with decay of learning rate alpha!

    if steam_tokens is None:
        steam_tokens = load_tokens()

    documents = list(read_corpus(steam_tokens))

    if model is None:
        model = doc2vec.Doc2Vec(
            documents,
        )  # training happens with 5 epochs (default) here

    start = time()
    model.train(documents, total_examples=len(documents), epochs=num_epochs)
    print('Elapsed time: {%.2f}' % (time() - start))

    model.save(get_doc_model_file_name())

    return model


def compute_similarity_using_doc2vec_model(
    query_app_id,
    steam_tokens=None,
    model=None,
    verbose=False,
    enforce_training=False,
    avoid_inference=False,
    num_items_displayed=10,
):
    if steam_tokens is None:
        steam_tokens = load_tokens()

    if model is None:
        try:
            print('Loading Doc2Vec model.')
            model = doc2vec.Doc2Vec.load(get_doc_model_file_name())

            if enforce_training:
                model = train_doc_model_on_steam_tokens(
                    model=model,
                    steam_tokens=steam_tokens,
                )

        except FileNotFoundError:
            print('Training Doc2Vec model from scratch.')
            model = train_doc_model_on_steam_tokens(
                model=None,
                steam_tokens=steam_tokens,
            )

    if avoid_inference:
        if verbose:
            print('Finding most similar documents based on the query appID.')
        # For games which are part of the training corpus, we do not need to call model.infer_vector()

        similarity_scores_as_tuples = model.docvecs.most_similar(
            positive=get_tag_prefix() + str(query_app_id),
            topn=num_items_displayed,
        )
    else:
        if verbose:
            print(
                'Finding most similar documents based on an inferred vector, which represents the query document.',
            )
        query = steam_tokens[query_app_id]
        # Caveat: « Subsequent calls to this function may infer different representations for the same document. »
        # Reference: https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec.infer_vector
        inferred_vector = model.infer_vector(query)
        similarity_scores_as_tuples = model.docvecs.most_similar([inferred_vector])

    similarity_scores = reformat_similarity_scores_for_doc2vec(
        similarity_scores_as_tuples,
    )
    print_most_similar_sentences(
        similarity_scores,
        num_items_displayed=num_items_displayed,
    )

    return similarity_scores


def check_analogy(model, pos, neg, num_items_displayed=10):
    similarity_scores_as_tuples = model.docvecs.most_similar(
        positive=[get_tag_prefix() + p for p in pos],
        negative=[get_tag_prefix() + n for n in neg],
        topn=num_items_displayed,
    )

    similarity_scores = reformat_similarity_scores_for_doc2vec(
        similarity_scores_as_tuples,
    )
    print_most_similar_sentences(similarity_scores, num_items_displayed)

    return


def apply_pipeline(
    train_from_scratch=True,
    avoid_inference=False,
    shuffle_corpus=True,
    include_genres=False,
    include_categories=True,
    include_app_ids=True,
    verbose=False,
):
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO,
    )

    game_names, game_tags = load_game_names(include_genres, include_categories)

    steam_tokens = load_tokens()

    documents = list(read_corpus(steam_tokens, game_tags, include_app_ids))

    if shuffle_corpus:
        # « Only if the training data has some existing clumping - like all the examples with certain words/topics are
        # stuck together at the top or bottom of the ordering - is native ordering likely to cause training problems.
        # And in that case, a single shuffle, before any training, should be enough to remove the clumping. »
        # Reference: https://stackoverflow.com/a/48080869
        random.shuffle(documents)

    if train_from_scratch:
        print('Creating a new Doc2Vec model from scratch.')
        model = doc2vec.Doc2Vec(
            documents,
            vector_size=100,
            window=5,
            min_count=5,
            epochs=20,
            workers=multiprocessing.cpu_count(),
        )

        # NB: Do not follow the piece of advice given in https://rare-technologies.com/doc2vec-tutorial/
        # « I have obtained better results by iterating over the data several times and either:
        #     1. randomizing the order of input sentences, or
        #     2. manually controlling the learning rate over the course of several iterations. »
        # Indeed, in my experience, this leads to buggy results. Moreover, this approach is not recommended according to
        # https://stackoverflow.com/a/48080869

        model.save(get_doc_model_file_name())
    else:
        print('Loading previous Doc2Vec model.')
        model = doc2vec.Doc2Vec.load(get_doc_model_file_name())

    # Test doc2vec

    if verbose:
        try:
            # Spelunky + (Slay the Spire) - (Dream Quest)
            check_analogy(model, pos=['239350', '646570'], neg=['557410'])
        except TypeError:
            pass

        try:
            # Half-Life + (Witcher 2) - (Witcher)
            check_analogy(model, pos=['70', '20920'], neg=['20900'])
        except TypeError:
            pass

        query_app_ids = [
            '620',
            '364470',
            '504230',
            '583950',
            '646570',
            '863550',
            '794600',
        ]

        for query_app_id in query_app_ids:
            print(f'Query appID: {query_app_id} ({game_names[query_app_id]})')
            compute_similarity_using_doc2vec_model(
                query_app_id,
                steam_tokens,
                model,
                avoid_inference=avoid_inference,
                num_items_displayed=10,
            )

        # Check the relevance of the corresponding word2vec
        for query_word in ['anime', 'fun', 'violent']:
            compute_similarity_using_word2vec_model(query_word, steam_tokens, model)

        entity = get_doc_model_entity(model)
        tag_entity = {tag for tag in entity if 'appID_' not in tag}

        print(tag_entity)

        query_tags = ['In-App Purchases', 'Free to Play', 'Violent', 'Early Access']

        for query_tag in tag_entity.intersection(query_tags):
            for query_app_id in query_app_ids:
                try:
                    sim = model.docvecs.similarity(
                        get_tag_prefix() + query_app_id,
                        query_tag,
                    )
                    print(
                        'Similarity = {:.0%} for tag {} vs. appID {} ({})'.format(
                            sim,
                            query_tag,
                            query_app_id,
                            game_names[query_app_id],
                        ),
                    )
                except KeyError:
                    pass

        num_items_displayed = 3
        for query_tag in tag_entity:
            print(f'\nTag: {query_tag}')
            similarity_scores_as_tuples = model.docvecs.most_similar(
                positive=query_tag,
                topn=num_items_displayed,
            )
            similarity_scores = reformat_similarity_scores_for_doc2vec(
                similarity_scores_as_tuples,
            )
            print_most_similar_sentences(
                similarity_scores,
                num_items_displayed=num_items_displayed,
            )

    # Top 100

    query_app_ids = load_benchmarked_app_ids(append_hard_coded_app_ids=True)

    num_neighbors = 10
    only_print_banners = True
    use_cosine_similarity = True

    label_database = np.array(model.docvecs.vectors_docs)
    doc_tags = list(model.docvecs.doctags.keys())

    init_indices = np.array(range(len(doc_tags)))
    bool_indices_to_remove = list(
        map(lambda x: not x.startswith(get_tag_prefix()), doc_tags),
    )
    indices_to_remove = init_indices[bool_indices_to_remove]
    label_database = np.delete(label_database, indices_to_remove, axis=0)

    app_ids = [
        int(doc_tag[len(get_tag_prefix()) :])
        for doc_tag in doc_tags
        if doc_tag.startswith(get_tag_prefix())
    ]

    knn = prepare_knn_search(
        label_database,
        use_cosine_similarity=use_cosine_similarity,
    )

    query_des = None
    for query_app_id in query_app_ids:
        if avoid_inference:
            inferred_vector = label_database[app_ids.index(query_app_id)]
        else:
            # From query appID to query feature vector
            query = steam_tokens[str(query_app_id)]
            # Caveat: « Subsequent calls to this function may infer different representations for the same document. »
            # Reference: https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec.infer_vector
            inferred_vector = model.infer_vector(query)

        if query_des is None:
            query_des = inferred_vector
        else:
            query_des = np.vstack((query_des, inferred_vector))

    # Matching of feature vectors
    matches = perform_knn_search_with_vectors_as_input(query_des, knn, num_neighbors)

    # From feature matches to appID matches
    matches_as_app_ids = transform_matches_to_app_ids(matches, app_ids)

    print_ranking(
        query_app_ids,
        matches_as_app_ids,
        num_elements_displayed=num_neighbors,
        only_print_banners=only_print_banners,
    )

    return


def get_doc_model_entity(model):
    # The equivalent of a vocabulary for a word model
    index2entity_set = set(model.docvecs.index2entity)
    return index2entity_set


if __name__ == '__main__':
    apply_pipeline(
        train_from_scratch=True,
        avoid_inference=False,
        shuffle_corpus=True,
        include_genres=False,
        include_categories=False,
        include_app_ids=True,
    )
