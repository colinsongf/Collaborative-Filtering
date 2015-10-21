__author__ = 'Ariel'

import time

import readHelper
import writeHelper
import util

start_time = time.time()
# read
M = readHelper.readTrain('train.csv')
#query,tuples = readHelper.readQuery("dev.csv")

k = 500
#func = 'dotp'
func = 'cosine'
#func_w = 'mean'
func_w = 'weight'
#pred = util.memCF(M, query, k, func, func_w)
#writeHelper.writePred('mem-10-dot-mean.txt', pred, tuples)
#writeHelper.writePred('mem-10-cosine-weight.txt', pred, tuples)
tuples = [] # input order maintenance
with open('dev.csv', 'r') as qu:
    for line in qu:
        t = line.split(",")
        tuples.append((int(t[1]), int(t[0])))
with open('mem-10-cosine-weight.txt','w') as f:
    for user,movie in tuples :
        #print user
        sim = util.similarity(M, user, func)
        temp = util.knn(sim, k)
        prediction = util.predict_new(M, temp, movie, sim, func_w)
        f.write('%d\n' % prediction)
# res = util.memcf(M, tuples, k, func, func_w)
# with open('mem-500-cosine-weight.txt', 'w') as f:
#     for x in res:
#         f.write('%d\n' % x)

print 'time: %s seconds' % (time.time()-start_time)