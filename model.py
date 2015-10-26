__author__ = 'Ariel'

import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix


def similarityPair(m, func):
    # m is the matrix
    if func == 'cosine':
        m = normalize(m)
    res = m.transpose().dot(m)
     # exclude query itself
    np.fill_diagonal(res,0)
    return res

def knn(similarity, k):
    return np.argpartition(-similarity, k)[:k]

def knnWeight(similarity, knn):
    return (similarity[[knn]]+1.)/2

def predict(m, knn, similarity, func):
    # column slicing and average
    if (func == 'weight') and (np.sum(knnWeight(similarity,knn)) != 0):
        res = np.average(m[:,knn], axis = 1, weights = knnWeight(similarity, knn))
    else:
        res = np.mean(m[:,knn], axis = 1)
    return res

def modelCF(m, tuples, k, func, func_w):
    res = [] # return the list of predictions given the query tuples
    m = np.asarray(m, order = 'F') # column-major order
    simPair = similarityPair(m, func)
    for pair in tuples:
        #print pair[0]
        sim = simPair[pair[0]]
        temp = knn(sim, k)
        prediction = predict(m, temp, sim, func_w)
        res.append(prediction[pair[1]] + 3)# plus 3
    return res


def pccModelCF(m, tuples, k, func, func_w):
    res = [] # return the list of predictions given the query tuples
    m = np.asarray(m, order = 'F') # column-major order
    norm, mu, std = pccU(m)

    simPair = similarityPair(norm, func)
    for pair in tuples:
        # print pair[0]
        sim = simPair[pair[0]]
        temp = knn(sim, k)
        prediction = predict(m, temp, sim, func_w)
        p = prediction[pair[1]] * std[pair[1]] + mu[pair[1]] + 3# plus 3
        #print p
        res.append(p)
    return res


def pccU(m):
    res = np.copy(m)
    mu = np.mean(m, axis = 1)
    std = np.std(m, axis = 1)
    for x in xrange(m.shape[0]):
        temp = m[x] - mu[x]
        if std[x] > 0:
            temp /= std[x]
        res[x] = temp
    return res, mu, std

def pccI(m):
    m = m.transpose()
    res = np.copy(m)
    mu = np.mean(m, axis = 1)
    std = np.std(m, axis = 1)
    for x in xrange(m.shape[0]):
        temp = m[x] - mu[x]
        if std[x] > 0:
            temp /= std[x]
        res[x] = temp
    return res.transpose(), mu, std

def pccUser(m):
    r = m.shape[0]
    c = m.shape[1]
    mu = np.mean(m, axis = 1)
    centered = m - np.repeat(mu, c).reshape(r,c)
    normalized = normalize(centered)
    return normalized

def pccItem(m):
    m = m.transpose()
    r = m.shape[0]
    c = m.shape[1]
    mu = np.mean(m, axis = 1)
    centered = m - np.repeat(mu, c).reshape(r,c)
    normalized = normalize(centered)
    return normalized.transpose()