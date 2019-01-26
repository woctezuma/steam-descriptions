import logging
import multiprocessing
from time import time

from gensim.models import doc2vec

from sentence_models import print_most_similar_sentences
from utils import load_tokens, load_game_names, get_doc_model_file_name
from word_model import compute_similarity_using_word2vec_model


def read_corpus(steam_tokens):
    for app_id, tokens in steam_tokens.items():
        yield doc2vec.TaggedDocument(tokens, [int(app_id)])


def reformat_similarity_scores_for_doc2vec(similarity_scores_as_tuples, game_names=None):
    if game_names is None:
        game_names, _ = load_game_names()

    dummy_app_ids = []

    similarity_scores = dict()
    for app_id, similarity_value in similarity_scores_as_tuples:
        # Remove dummy appIDs
        if str(app_id) in game_names:
            similarity_scores[str(app_id)] = similarity_value
        else:
            dummy_app_ids.append(app_id)

    print('Dummy appIDs: {}'.format(dummy_app_ids))

    return similarity_scores


def train_doc_model_on_steam_tokens(model=None, steam_tokens=None, num_epochs=10):
    # You do not want to perform training this way, because training already happened when initializating the model
    # with Doc2Vec(documents). Moreover, calling train() several times messes with decay of learning rate alpha!

    if steam_tokens is None:
        steam_tokens = load_tokens()

    documents = list(read_corpus(steam_tokens))

    if model is None:
        model = doc2vec.Doc2Vec(documents)  # training happens with 5 epochs (default) here

    start = time()
    model.train(documents, total_examples=len(documents), epochs=num_epochs)
    print('Elapsed time: {%.2f}' % (time() - start))

    model.save(get_doc_model_file_name())

    return model


def compute_similarity_using_doc2vec_model(query_app_id, steam_tokens=None, model=None,
                                           enforce_training=False, avoid_inference=False, num_items_displayed=10):
    if steam_tokens is None:
        steam_tokens = load_tokens()

    if model is None:
        try:
            print('Loading Doc2Vec model.')
            model = doc2vec.Doc2Vec.load(get_doc_model_file_name())

            if enforce_training:
                model = train_doc_model_on_steam_tokens(model=model, steam_tokens=steam_tokens)

        except FileNotFoundError:
            print('Training Doc2Vec model from scratch.')
            model = train_doc_model_on_steam_tokens(model=None, steam_tokens=steam_tokens)

    if avoid_inference:
        print('Finding most similar documents based on the query appID.')
        # For games which are part of the training corpus, we do not need to call model.infer_vector()

        num_items_retrieved = max(4 * num_items_displayed, 20)
        similarity_scores_as_tuples = model.docvecs.most_similar(positive=int(query_app_id), topn=num_items_retrieved)

        # Hack for display with print_most_similar_sentences():
        # if model.docvecs.most_similar() is called with an integer doctag found in the training set,
        # then the doctag is not returned! So, we add it to the list of tuples for later display!
        perfect_similarity_score = 1.0
        if all(query_app_id != app_id for (app_id, similarity_value) in similarity_scores_as_tuples):
            similarity_scores_as_tuples.append((query_app_id, perfect_similarity_score))
    else:
        print('Finding most similar documents based on an inferred vector, which represents the query document.')
        query = steam_tokens[query_app_id]
        # Caveat: « Subsequent calls to this function may infer different representations for the same document. »
        # Reference: https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec.infer_vector
        inferred_vector = model.infer_vector(query)
        similarity_scores_as_tuples = model.docvecs.most_similar([inferred_vector])

    similarity_scores = reformat_similarity_scores_for_doc2vec(similarity_scores_as_tuples)
    print_most_similar_sentences(similarity_scores, num_items_displayed=num_items_displayed)

    return similarity_scores


def check_analogy(model, pos, neg):
    similarity_scores_as_tuples = model.docvecs.most_similar(positive=pos, negative=neg, topn=20)

    similarity_scores = reformat_similarity_scores_for_doc2vec(similarity_scores_as_tuples)
    print_most_similar_sentences(similarity_scores, num_items_displayed=3, is_query_included=False)

    return


if __name__ == '__main__':
    train_from_scratch = False

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    steam_tokens = load_tokens()

    if train_from_scratch:
        print('Creating a new Doc2Vec model from scratch.')
        documents = list(read_corpus(steam_tokens))
        model = doc2vec.Doc2Vec(documents,
                                num_epochs=20,
                                workers=multiprocessing.cpu_count())
    else:
        print('Loading previous Doc2Vec model.')
        model = doc2vec.Doc2Vec.load(get_doc_model_file_name())

    # Test doc2vec
    for query_app_id in ['10', '620', '105600', '264710', '292030', '294100', '364470', '504230', '519860', '531640',
                         '560130', '582010', '583950', '588650', '590380', '620980', '638970', '644560', '646570',
                         '653530', '683320', '698780', '731490', '742120', '812140', '863550', '973760']:
        compute_similarity_using_doc2vec_model(query_app_id, steam_tokens, model, avoid_inference=False)

    check_analogy(model, pos=[239350, 646570], neg=[557410])  # Spelunky + (Slay the Spire) - (Dream Quest)
    check_analogy(model, pos=[70, 20920], neg=[20900])  # Half-Life + (Witcher 2) - (Witcher)

    # Check the relevance of the corresponding word2vec
    for query_word in ['anime', 'fun', 'violent']:
        compute_similarity_using_word2vec_model(query_word, steam_tokens, model)
