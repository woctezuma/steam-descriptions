# Objective: load data, tokenize & embed sentences with the 'flair' library @ https://github.com/zalandoresearch/flair/

import json
from time import time

from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence

from utils import load_game_names, load_tokens, get_embedding_file_name


def compute_steam_embeddings(steam_tokens=None, save_to_disk=False):
    if steam_tokens is None:
        steam_tokens = load_tokens()

    game_names, _ = load_game_names(include_genres=False, include_categories=False)

    counter = 0
    num_games = len(steam_tokens)

    steam_embeddings = {}

    # initialize the word embeddings
    glove_embedding = WordEmbeddings('glove')
    flair_embedding_forward = FlairEmbeddings('news-forward')
    flair_embedding_backward = FlairEmbeddings('news-backward')

    # initialize the document embeddings, mode = mean
    document_embeddings = DocumentPoolEmbeddings([glove_embedding,
                                                  flair_embedding_backward,
                                                  flair_embedding_forward])

    start = time()

    for app_id in steam_tokens:
        game_tokens = steam_tokens[app_id]
        game_name = game_names[app_id]
        counter += 1

        if (counter % 1) == 0:
            if counter > 1:
                print('Elapsed time: {%.2f}' % (time() - start))
            start = time()
            print('[{}/{}] appID = {} ({})'.format(counter, num_games, app_id, game_name))

        game_str = ' '.join(game_tokens)

        # Tokenize the text with the 'flair' library
        sentence = Sentence(game_str)

        # embed the sentence with our document embedding
        document_embeddings.embed(sentence)

        steam_embeddings[app_id] = sentence.get_embedding()

    if save_to_disk:
        with open(get_embedding_file_name(), 'w') as f:
            json.dump(steam_embeddings, f)

    return steam_embeddings


if __name__ == '__main__':
    steam_embeddings = compute_steam_embeddings(save_to_disk=True)
