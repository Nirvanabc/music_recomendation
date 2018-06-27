import struct
import re
import numpy as np
from numpy.random import normal
from random import shuffle
import pickle
from constants import *
import gensim


def normalize(vec):
    total = sum(vec)
    return list(map(lambda x: x/total, vec))

## FIXME! normalize:
# maybe it should be np.random.uniform(0, 1, (iter_num, 3))

def word2vec(word, vec_size):
    try:
        result = dictionary[word]
        return result
    except KeyError:
        new_vec = normalize(normal(size = vec_size))
        result = dictionary[word] = new_vec
        return result


def corpora2vec(corpora, vec_size):
    result = []
    for sent in corpora:
        curr = []
        for word in sent:
            curr.append(word2vec(word, vec_size))
            # to test without softlink_ru
            # curr.append(normalize(normal(size = vec_size)))
        result.append(curr)
    return result


def padd_sent(sent, vec_size, sent_size):
    res_sent = np.zeros([sent_size, vec_size])
    if len(sent) <= sent_size:
        res_sent[sent_size - len(sent):] = sent
    else: res_sent = sent[:sent_size]
    return res_sent


def padd_corpora(corpora, vec_size, sent_size):
    res_corpora = []
    for sent in corpora:
        res_corpora.append(padd_sent(sent,         \
                                     vec_size,     \
                                     sent_size))
    return res_corpora


def del_empty(corpora):
    return list(filter(lambda x: x!=[], corpora))


def prepare_corpora(corpora, vec_size, \
                    sent_size):
    '''
    takes a batch and prepare it
    '''
    # corpora = del_empty(corpora)
    vec_dictionary = corpora2vec(corpora, vec_size)
    vec_dictionary = padd_corpora(vec_dictionary, \
                                  vec_size,       \
                                  sent_size)
    return vec_dictionary    


def rand(vec_size):
    return normalize(normal(size = vec_size))


def store_data(data, file_to_store):
    f = open(file_to_store, 'wb')
    pickle.dump(data, f)
    f.close()


def next_batch(corpora, n, vec_size):
    batch = []
    labels = []
    for _ in range(n):
        sent = corpora.readline()
        if len(sent) == 0:
            return 0
        sent = sent.split()
        if sent[:-1] != []:
            labels.append(int(sent[-1]))
            batch.append(sent[:-1])
    batch = prepare_corpora(batch, vec_size, sent_size)
    labels = [[1-labels[i], \
               labels[i]] for i in range(len(labels))]
    batch = [batch, labels]
    return batch

# # building a dictionary
# model = gensim.models.KeyedVectors.load_word2vec_format(
#     './softlink_en_big', binary=True)
# dictionary = {}
# for key in model.vocab:
#     if str.isalpha(key):
#         dictionary[key.lower()] = model.wv[key]

dictionary, vec_size = get_dict(en_dict_source)
