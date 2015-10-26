__author__ = 'Ariel'

import numpy as np
from scipy.sparse import coo_matrix

# read training file, (movie, user, rating) into a user-movie matrix
def readTrainModel(file):
    user = []
    movie = []
    rating = []
    with open(file, 'r') as tr:
        for line in tr:
            t = line.split(",")
            user.append(int(t[1]))
            movie.append(int(t[0]))
            rating.append(int(t[2])-3) # pre-process here, option 2

    m = coo_matrix((rating, (user, movie)), dtype = float)
    m = m.toarray()
    return m

def readTrainMemory(file):
    user = []
    movie = []
    rating = []
    with open(file, 'r') as tr:
        for line in tr:
            t = line.split(",")
            user.append(int(t[1]))
            movie.append(int(t[0]))
            rating.append(int(t[2])-3) # pre-process here, option 2

    m = coo_matrix((rating, (user, movie)), dtype = float)
    m = m.tocsr()
    return m

# read query file into a user-movies dictionary
def readQueryMemory(file):
    query = {} # user as key, list of movies as value
    tuples = [] # input order maintenance
    with open(file, 'r') as qu:
        for line in qu:
            t = line.split(",")
            tuples.append((int(t[1]), int(t[0])))
            try:
                query[int(t[1])].append(int(t[0]))
            except:
                query[int(t[1])] = [int(t[0])]
    return query, tuples

def readQueryModel(file):
    tuples = []
    with open(file, 'r')as qu:
        for line in qu:
            t = line.split(",")
            tuples.append((int(t[0]),int(t[1])))
    return tuples
