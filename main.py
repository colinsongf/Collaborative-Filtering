__author__ = 'Ariel'

import time

import readHelper
import writeHelper
import util

start_time = time.time()
# read
M = readHelper.readTrain('train.csv')
query,tuples = readHelper.readQuery("dev.csv")

k = 500
#func = 'dotp'
func = 'cosine'
#func_w = 'mean'
func_w = 'weight'
pred = util.memCF(M, query, k, func, func_w)
#writeHelper.writePred('mem-10-dot-mean.txt', pred, tuples)
writeHelper.writePred('mem-10-cosine-weight.txt', pred, tuples)



print 'time: %s seconds' % (time.time()-start_time)