# Reference: https://github.com/PrincetonML/SIF

import numpy as np

import SIF_embedding


def weighted_average_sim_rmpc(word_embed, x1, x2, w1, w2, params):
    """
    Compute scores between pairs of sentences with weighted average + removing projection on the 1st principal component
    :param word_embed: We[i,:] is the vector for word i
    :param x1: x1[i, :] are the indices of the words in the first sentence in pair i
    :param x2: x2[i, :] are the indices of the words in the second sentence in pair i
    :param w1: w1[i, :] are the weights for the words in the first sentence in pair i
    :param w2: w2[i, :] are the weights for the words in the first sentence in pair i
    :param params: if params.rmpc >0, remove the projections of the sentence embeddings to their 1st principal component
    :return: scores, scores[i] is the matching score of the pair i
    """
    emb1 = SIF_embedding.sif_embedding(word_embed, x1, w1, params)
    emb2 = SIF_embedding.sif_embedding(word_embed, x2, w2, params)

    inn = (emb1 * emb2).sum(axis=1)
    emb1norm = np.sqrt((emb1 * emb1).sum(axis=1))
    emb2norm = np.sqrt((emb2 * emb2).sum(axis=1))
    scores = inn / emb1norm / emb2norm
    return scores
