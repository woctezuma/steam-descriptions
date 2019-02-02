# Code from: Sanjeev Arora, Yingyu Liang, Tengyu Ma, "A Simple but Tough-to-Beat Baseline for Sentence Embeddings", 2016
# Reference: https://github.com/PrincetonML/SIF

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
    Get SIF embedding, i.e. emb[i,:] is the embedding for sentence i.
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
