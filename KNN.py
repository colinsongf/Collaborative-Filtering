__author__ = 'Ariel'

import numpy as np
from scipy.spatial.distance import cdist


# for matrix with query row to get knn index and value
def knn(m, q, k, func):
    # m is the user-movie-rating matrix, q is the query
    # k is the number returned, func is the similarity measure function
    if func == 'dotp':
        res = m.dot(q)
    if func == 'cosine':
        q = q.reshape([1, q.shape[0]])
        res = 1 - cdist(m, q, 'cosine') # distance to similarity
        res = res.reshape([res.shape[0], ])
        res[np.isnan(res)] = 0 # not a number in cosine similarity
    # return the list of index and value of K nearest neighbor
    idx = np.argsort(res)[::-1][1:k] # exclude itself here
    # idx = res[1:k]
    val = res[idx]
    return  idx, val

# for combination/prediction using knn
def combine(m, knn_idx, knn_value, func):
    # m is the user-movie-rating matrix,
    if func == 'mean':
        res =  np.mean(m[knn_idx], axis = 0)
    if func == 'weights':
        # knn_value is weight, which is cosine similarity
        #weight = (np.array(knn_value)+1)/2
        #res = np.average(m[knn_idx], axis = 0, weights = weight)
        res = np.average(m[knn_idx], axis = 0, weights = knn_value)
    return 3 + np.rint(res)# round to nearest integer and add 3

# pairwise similarity for matrix
def similarity(m, func):
    if func == 'dotp':
        sim = m.dot(m.transpose())
    if func == 'cosine':
        sim = 1 - cdist(m, m, 'cosine')
        sim[np.isnan(sim)] = 0 # not a number in cosine similarity
    return sim
