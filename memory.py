__author__ = 'Ariel'

import numpy as np
from sklearn.preprocessing import normalize
from numpy.linalg import norm
from scipy.sparse import csr_matrix


def similarity(m, q, func):
    # m is the matrix and q is the query row index
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

def pccMemoryCF(m, query, k, func, func_w):
    res = {}
    norm = pccUser(m)
    for user in query.keys():
        #print user
        sim = similarity(norm, user, func)
        temp = knn(sim, k)
        prediction = predict(m, temp, sim, func_w)
        for movie in query[user]:
            if prediction[movie] < 1:
                res[user, movie] = 1
            elif prediction[movie] > 5:
                prediction[movie] = 5
            else:
                res[user, movie] = prediction[movie]
    return res


def pccI(m):
    m = csr_matrix(m.transpose())
    for x in xrange(m.shape[0]):
        if m[x].size:
            mu = np.mean(m[x].data)
            std = np.std(m[x].data)
            m.data[m.indptr[x]:m.indptr[x+1]] -= mu
            if std > 0:
                m.data[m.indptr[x]:m.indptr[x+1]] /= std

    return csr_matrix(m.transpose())

def pccU(m):

    for x in xrange(m.shape[0]):
        if m[x].size:
            mu = np.mean(m[x].data)
            std = np.std(m[x].data)
            m.data[m.indptr[x]:m.indptr[x+1]] -= mu
            if std > 0:
                m.data[m.indptr[x]:m.indptr[x+1]] /= std

    return m

def pccUser(m):

    for x in xrange(m.shape[0]):
        if m[x].size:
            mu = np.mean(m[x].data)
            n = norm(m[x].data-mu)
            m.data[m.indptr[x]:m.indptr[x+1]] -= mu
            if n > 0:
                m.data[m.indptr[x]:m.indptr[x+1]] /= n

    return m