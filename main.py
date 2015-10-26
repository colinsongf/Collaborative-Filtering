__author__ = 'Ariel'

import time

import readHelper
import writeHelper
import model
import memory

start_time = time.time()
# read
M = readHelper.readTrain('train.csv')
#query,tuples = readHelper.readQueryMemory("dev.csv")

tuples = readHelper.readQueryModel("dev.csv")
k = 10
#func = 'dotp'
func = 'cosine'
#func_w = 'mean'
func_w = 'weight'
pred = model.modelCF(M, tuples, k, func, func_w)
writeHelper.writePredModel('mod-10-cosine-weight.txt', pred)
#writeHelper.writePredMemory('mem-10-dot-mean.txt', pred, tuples)


print 'time: %s seconds' % (time.time()-start_time)