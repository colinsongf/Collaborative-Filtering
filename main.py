__author__ = 'Ariel'


import numpy as np
import time
import readHelper
import writeHelper
import CF
import KNN

start_time = time.time()
# read
M = readHelper.readTrain('train.csv')
#query,tuples = readHelper.readQuery("dev.csv")

k = 10
func = 'dotp'
func_com = 'mean'
# # memory-based user-user similarity
# pred = CF.memoryCF(M, query, k, func, func_com)
# # write
# writeHelper.writePred('mem-10-dot-mean.txt', pred, tuples)

# tuples = [] # input order maintenance
# with open('dev.csv', 'r') as qu:
#     for line in qu:
#         t = line.split(",")
#         tuples.append((int(t[1]), int(t[0])))
#
# res = []
#
# i = 0
# for (user,movie) in tuples:
#         print i
#         i += 1
#         knn_idx, knn_value = KNN.knn(M, M[user], k, func)
#         prediction = KNN.combine(M, knn_idx, knn_value, func_com)
#         res.append(prediction[movie])
#
# with open('test.txt','w') as f:
#     for x in res:
#         f.write('%d\n' % x)
# print 'time: %s seconds' % (time.time()-start_time)