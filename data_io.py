# Code from: Sanjeev Arora, Yingyu Liang, Tengyu Ma, "A Simple but Tough-to-Beat Baseline for Sentence Embeddings", 2016
# Reference: https://github.com/PrincetonML/SIF


import numpy as np


def get_word_map(text_file):
    words = {}
    word_embedding = []
    f = open(text_file, 'r')
    lines = f.readlines()
    for (n, i) in enumerate(lines):
        i = i.split()
        j = 1
        v = []
        while j < len(i):
            v.append(float(i[j]))
            j += 1
        words[i[0]] = n
        word_embedding.append(v)
    return words, np.array(word_embedding)


def get_word_weight(weight_file, a=1e-3):
    # word2weight['str'] is the weight for the word 'str'
    if a <= 0:  # when the parameter makes no sense, use unweighted
        a = 1.0

    word2weight = {}
    with open(weight_file) as f:
        lines = f.readlines()
    n = 0
    for i in lines:
        i = i.strip()
        if len(i) > 0:
            i = i.split()
            if len(i) == 2:
                word2weight[i[0]] = float(i[1])
                n += float(i[1])
            else:
                print(i)
    for key, value in word2weight.items():
        word2weight[key] = a / (a + value / n)
    return word2weight


def get_seq(p1, words):
    p1 = p1.split()
    x1 = []
    for i in p1:
        x1.append(look_up_idx(words, i))
    return x1


def look_up_idx(words, w):
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#", "")
    if w in words:
        return words[w]
    else:
        return len(words) - 1


def get_weight(words, word2weight):
    # weight4ind[i] is the weight for the i-th word
    weight4ind = {}
    for word, ind in words.iteritems():
        if word in word2weight:
            weight4ind[ind] = word2weight[word]
        else:
            weight4ind[ind] = 1.0
    return weight4ind


def prepare_data(list_of_seqs):
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    max_length = np.max(lengths)
    x = np.zeros((n_samples, max_length)).astype('int32')
    x_mask = np.zeros((n_samples, max_length)).astype('float32')
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
    x_mask = np.asarray(x_mask, dtype='float32')
    return x, x_mask


def sentences2idx(sentences, words):
    """
    Given a list of sentences, output array of word indices that can be fed into the algorithms.
    :param sentences: a list of sentences
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1.    x1[i, :] is the word indices in sentence i,
                        m1[i,:] is the binary mask for sentence i (0 means that there is no word at the location)
    """
    seq1 = []
    for i in sentences:
        seq1.append(get_seq(i, words))
    x1, m1 = prepare_data(seq1)
    return x1, m1


def seq2weight(seq, mask, weight4ind):
    # get word weights
    weight = np.zeros(seq.shape).astype('float32')
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            if mask[i, j] > 0 and seq[i, j] >= 0:
                weight[i, j] = weight4ind[seq[i, j]]
    weight = np.asarray(weight, dtype='float32')
    return weight
