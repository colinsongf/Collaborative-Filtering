__author__ = 'Ariel'

import random
import numpy as np
from sklearn.preprocessing import normalize

def BiClustering(data, k1, k2):
    word2Dcluster, doc2D = Kmean(data, k1)
    iter = 0
    while iter < 20:
        print iter
        _, word2W = Kmean(word2Dcluster.transpose(), k2)
        doc2Wcluster = np.zeros((data.shape[0], len(word2W)))
        for i in word2W.keys():
            doc2Wcluster[:,i] = np.mean(data.transpose()[word2W[i]], axis = 0)
        _, doc2D = Kmean(doc2Wcluster, k1)
        word2Dcluster = np.zeros((len(doc2D), data.shape[1]))
        for i in doc2D.keys():
            word2Dcluster[i] = np.mean(data[doc2D[i]], axis = 0)
        iter += 1
    return (doc2D, word2W, doc2Wcluster, word2Dcluster)


def Kmean(data, k):
    "K-means clustering on data matrix, row vectors based"
    # Initialization
    seed = random.sample(xrange(data.shape[0]), k)
    centers = data[seed]
    newAssign = [0]*data.shape[0]
    assign = [1]*data.shape[0]
    # Iteration
    iter = 0
    while newAssign != assign and iter < 30:
        assign = newAssign
        clusters, newAssign = getCluster(data, centers, seed)
        centers = getCenter(data, clusters)
        iter += 1
        #print iter
    return (np.array(centers), clusters)


def getCluster(data, centers, seed):
    clusters = {}
    assign = np.argmax(normalize(data).dot(normalize(centers.transpose())), axis=1)
    # maintain k size
    for x, y in enumerate(assign):
        if x in seed:
            assign[x] = seed.index(x)
    for (x, y) in zip(xrange(assign.shape[0]), assign):
        try:
            clusters[y].append(x)
        except:
            clusters[y] = [x]
    return (clusters, assign.tolist())


def getCenter(data, clusters):
    newCenters = np.zeros((len(clusters), data.shape[1]))
    idx = 0
    for i in clusters.keys():
        newCenters[idx] = np.mean(data[clusters[i]], axis = 0)
        idx += 1
    return newCenters

