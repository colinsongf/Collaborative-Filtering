__author__ = 'Ariel'

import numpy as np
from sklearn.preprocessing import normalize


def similarityPair(m, func):
    # m is the matrix
    # for model based CF
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
    return 3 + res # plus 3

def modelCF(m, tuples, k, func, func_w):
    res = [] # return the list of predictions given the query tuples
    simPair = similarityPair(m, func)
    m = np.asarray(m, order = 'F') # column-major order
    for pair in tuples:
        #print pair[0]
        sim = simPair[pair[0]]
        temp = knn(sim, k)
        prediction = predict(m, temp, sim, func_w)
        res.append(prediction[pair[1]])
    return res