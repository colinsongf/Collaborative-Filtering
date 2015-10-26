__author__ = 'Ariel'

import numpy as np
from scipy.sparse import csr_matrix

import readHelper
import model

# training set exploration

#np.seterr(divide='ignore', invalid='ignore')
user = []
movie = []
rating = []
with open('train.csv') as dev:
    for line in dev:
        t = line.split(",")
        user.append(int(t[1]))
        movie.append(int(t[0]))
        rating.append(int(t[2]))

m = csr_matrix((rating, (user, movie)))

print m.nnz
print m.shape

r = np.array(rating)
print np.where(r == 1)[0].shape
print np.where(r == 3)[0].shape
print np.where(r == 5)[0].shape
print np.average(r)

M = m.toarray()

u4321 = M[4321]

print u4321.nonzero()[0].size
print np.where(u4321 == 1)[0].shape
print np.where(u4321 == 3)[0].shape
print np.where(u4321 == 5)[0].shape
print np.average(u4321[u4321.nonzero()[0].tolist()])

m3 = M[:,3]

print m3.nonzero()[0].size
print np.where(m3 == 1)[0].shape
print np.where(m3 == 3)[0].shape
print np.where(m3 == 5)[0].shape
print np.average(m3[m3.nonzero()[0].tolist()])

M = readHelper.readTrain('train.csv')

sim = model.similarity(M, 4321, 'dotp')
print model.knn(sim,5)
sim = model.similarity(M, 4321, 'cosine')
print model.knn(sim,5)

sim = model.similarity(M.transpose(), 3, 'dotp')
print model.knn(sim,5)
sim = model.similarity(M.transpose(), 3, 'cosine')
print model.knn(sim,5)

