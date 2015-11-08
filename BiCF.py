__author__ = 'Ariel'
import pickle
import numpy as np
import memory
import model


def bi_item(m, tuples, k, func, func_w):
    # with open('u2U.txt', 'r') as f1:
    #     u2U = pickle.loads(f1.read())
    # with open('m2Ucluster.txt', 'r') as f4:
    #     m2Ucluster = pickle.loads(f4.read())
    with open('m2M.txt', 'r') as f1:
        m2M = pickle.loads(f1.read())
    with open('u2Mcluster.txt', 'r') as f2:
        u2Mcluster = pickle.loads(f2.read())

    res = [] # return the list of predictions given the query tuples
    # m = np.asarray(m, order = 'F') # column-major order
    u2Mcluster = np.asarray(u2Mcluster, order = 'F')
    simPair = model.similarityPair(u2Mcluster , func)
    for pair in tuples:
        #print pair[0]
        c = findCentroid(m2M, pair[0])
        sim = simPair[c]
        temp = model.knn(sim, k)
        prediction = model.predict(u2Mcluster, temp, sim, func_w)
        pred = prediction[pair[1]] + 3
        if pred > 5:
            pred = 5
        elif pred < 1:
            pred = 1
        res.append(pred)# plus 3
        #print pred
    return res

def bi_user(m, tuples, k, func, func_w):
    with open('u2U.txt', 'r') as f1:
        u2U = pickle.loads(f1.read())
    with open('m2Ucluster.txt', 'r') as f2:
        m2Ucluster = pickle.loads(f2.read())
    m2Ucluster = m2Ucluster.transpose()

    res = [] # return the list of predictions given the query tuples
    m2Ucluster = np.asarray(m2Ucluster, order = 'F')
    simPair = model.similarityPair(m2Ucluster , func)
    for pair in tuples:
        #print pair[0]
        c = findCentroid(u2U, pair[1])
        sim = simPair[c]
        temp = model.knn(sim, k)
        prediction = model.predict(m2Ucluster, temp, sim, func_w)
        pred = prediction[pair[0]] + 3
        if pred > 5:
            pred = 5
        elif pred < 1:
            pred = 1
        res.append(pred)# plus 3
        #print pred
    return res

def findCentroid(dict, target):
    for key,values in dict.iteritems():
        if  target in values:
            return key
