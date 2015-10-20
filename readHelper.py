__author__ = 'Ariel'

import numpy as np
from scipy.sparse import coo_matrix


def readTrain(file):
    user = []
    movie = []
    rating = []
    with open(file, 'r') as tr:
        for line in tr:
            t = line.split(",")
            user.append(int(t[1]))
            movie.append(int(t[0]))
            rating.append(int(t[2])-3) # pre-process here, option 2

    m = coo_matrix((rating, (user, movie)))
    M = m.toarray()
    return M

def readQuery(file):
    query = {}
    with open(file, 'r') as qu:
        for line in qu:
            t = line.split(",")
            try:
                query[int(t[1])].append(int(t[0]))
            except:
                query[int(t[1])] = [int(t[0])]
    return query