__author__ = 'Ariel'

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

def similarity(m, q, func):
    # m is the matrix and q is the query row index
    if func == 'cosine':
        m = normalize(m)
    res = m.dot(m[q].transpose()).toarray().flatten() # flatten for partition
    # exclude query itself
    res[q] = min(res)
    return res

def knn(similarity, k):
    return np.argpartition(-similarity, k)[:k]

def knnWeight(similarity, knn):
    return similarity[[knn]]

def predict(m, knn, similarity, func):
    if (func == 'weight') and (np.sum(knnWeight(similarity,knn)) != 0):
        res = np.average(m[knn].toarray(), axis = 0, weights = knnWeight(similarity, knn))
    else:
        res =  np.mean(m[knn].toarray(), axis = 0)
    return 3 + np.rint(res) # nearest integer and plus 3

def memCF(m, query, k, func, func_w):
    res = {}
    for user in query.keys():
        print user
        sim = similarity(m, user, func)
        temp = knn(sim, k)
        prediction = predict(m, temp, sim, func_w)
        for movie in query[user]:
            res[user, movie] = prediction[movie]
    return res
