__author__ = 'Ariel'

import numpy as np
from sklearn.preprocessing import normalize

def similarity(m, q, func):
    # m is the matrix and q is the query row index
    # for memory based CF
    if func == 'cosine':
        m = normalize(m)
    res = m.dot(m[q].transpose())
    res[q] = 0 # exclude query itself
    return res

def similarityPair(m, func):
    # m is the matrix
    # for model based CF
    if func == 'cosine':
        m = normalize(m)
    res = m.transpose().dot(m)
     # exclude query itself

    return res

def knn(similarity, k):
    return np.argpartition(-similarity, k)[:k]

def knnWeight(similarity, knn):
    return similarity[[knn]]

def predict(m, knn, similarity, func):
    if (func == 'weight') and (np.sum(knnWeight(similarity,knn)) != 0):
        res = np.average(m[knn], axis = 0, weights = knnWeight(similarity, knn))
    else:
        res = np.mean(m[knn], axis = 0)
    return 3 + res # nearest integer and plus 3

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

def modelCF(m, tuples, k, func, func_w):
    res = [] # return the list of predictions given the query tuples
    simPair = similarityPair(m, func)
    for pair in tuples:
        #print pair[0]
        sim = simPair[pair[0]]
        temp = knn(sim, k)
        prediction = predict(m, temp, sim, func_w)
        res.append(prediction)
    return res