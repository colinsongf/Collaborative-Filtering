__author__ = 'Ariel'

import numpy as np
from sklearn.preprocessing import normalize

def similarity(m, q, func):
    # m is the matrix and q is the query row index
    # for memory based CF
    if func == 'cosine':
        m = normalize(m)
    res = m.dot(m[q].transpose()).toarray().flatten()
    res[q] = 0 # exclude query itself
    return res

def knn(similarity, k):
    return np.argpartition(-similarity, k)[:k]

def knnWeight(similarity, knn):
    return (similarity[[knn]]+1.)/2

def predict(m, knn, similarity, func):
     # row slicing and average
    if (func == 'weight') and (np.sum(knnWeight(similarity,knn)) != 0):
        res = np.average(m[knn].toarray(), axis = 0, weights = knnWeight(similarity, knn))
    else:
        res = np.mean(m[knn].toarray(), axis = 0)
    return 3 + res # plus 3

def memoryCF(m, query, k, func, func_w):
    res = {} # return the dictionary {u,m} = prediction for all the queries
    for user in query.keys():
        #print user
        sim = similarity(m, user, func)
        temp = knn(sim, k)
        prediction = predict(m, temp, sim, func_w)
        for movie in query[user]:
            res[user, movie] = prediction[movie]
    return res
