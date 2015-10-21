__author__ = 'Ariel'

import numpy as np
from scipy.sparse import coo_matrix,csr_matrix
import readHelper
import KNN

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

q1 = M[4321]
print KNN.knn(M, q1, 6, 'dotp')
print KNN.knn(M, q1, 6, 'cosine')


q2 = M[:,3]
print KNN.knn(M.transpose(), q2, 6, 'dotp')
print KNN.knn(M.transpose(), q2, 6, 'cosine')

