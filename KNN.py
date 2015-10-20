__author__ = 'Ariel'

import numpy as np
from scipy.spatial.distance import cdist

# for matrix with query row
def knn(m, q, k, func):
    # m is the user-movie-rating matrix, q is the query
    # k is the number returned, func is the similarity measure function

    if func == 'dotp':
        res = m.dot(q)
    if func == 'cosine':
        q = q.reshape([1, q.shape[0]])
        res = 1 - cdist(m, q, 'cosine')# distance to similarity
        res = res.reshape([res.shape[0], ])
        res[np.isnan(res)] = 0 # not a number in cosine similarity
    # return the list of index of K nearest neighbor
    return np.argsort(res)[::-1][1:k]# exclude itself here

# pairwise similarity for matrix
def similarity(m, func):
    if func == 'dotp':
        sim = m.dot(m.transpose())
    if func == 'cosine':
        sim = 1 - cdist(m, m, 'cosine')
        sim[np.isnan(sim)] = 0 # not a number in cosine similarity
    return sim