# Reference: https://github.com/PrincetonML/SIF

from collections import namedtuple

import numpy as np
from sklearn.decomposition import TruncatedSVD


def get_weighted_average(word_embed, x, w):
    """
    Compute the weighted average vectors
    :param word_embed: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, word_embed.shape[1]))
    for i in range(n_samples):
        emb[i, :] = w[i, :].dot(word_embed[x[i, :], :]) / np.count_nonzero(w[i, :])
    return emb


def compute_pc(x, npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param x: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(x)
    return svd.components_


def remove_pc(x, npc=1):
    """
    Remove the projection on the principal components
    :param x: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(x, npc)
    if npc == 1:
        xx = x - x.dot(pc.transpose()) * pc
    else:
        xx = x - x.dot(pc.transpose()).dot(pc)
    return xx


def sif_embedding(word_embed, x, w, params):
    """
    Compute scores between pairs of sentences using weighted average + removing projection on 1st principal component
    :param word_embed: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in the i-th sentence
    :param w: w[i, :] are the weights for the words in the i-th sentence
    :param params: if params.rmpc >0, remove projections of the sentence embeddings to their 1st principal component.
    :return: emb, emb[i, :] is the embedding for sentence i
    """
    emb = get_weighted_average(word_embed, x, w)
    if params.rmpc > 0:
        emb = remove_pc(emb, params.rmpc)
    return emb


if __name__ == '__main__':
    import data_io

    # input
    word_file = '../data/glove.840B.300d.txt'  # word vector file, can be downloaded from GloVe website
    weight_file = '../auxiliary_data/enwiki_vocab_min200.txt'  # each line is a word and its frequency
    weight_parameter = 1e-3  # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
    sentences = ['this is an example sentence', 'this is another sentence that is slightly longer']

    # load word vectors
    (words, We) = data_io.get_word_map(word_file)
    # load word weights
    word2weight = data_io.get_word_weight(weight_file,
                                          weight_parameter)  # word2weight['str'] is the weight for the word 'str'
    weight4ind = data_io.get_weight(words, word2weight)  # weight4ind[i] is the weight for the i-th word
    # load sentences
    x, m, _ = data_io.sentences2idx(sentences,
                                    words)  # x is the array of word indices,
    # m is the binary mask indicating whether there is a word in that location
    w = data_io.seq2weight(x, m, weight4ind)  # get word weights

    # set parameters
    Params = namedtuple('Params', 'rmpc')
    params = Params(1)  # number of principal components to remove in SIF weighting scheme
    # get SIF embedding
    embedding = sif_embedding(We, x, w, params)  # embedding[i,:] is the embedding for sentence i
