__author__ = 'Ariel'

import time
import argparse
import readHelper
import writeHelper
import model
import memory

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description = "collaborative filtering")
    parser.add_argument("-m", help = "memory-based or model collaborative filtering", choices = ['memory','model'])
    parser.add_argument("-k", help = "number of k nearest neighborhood", type = int, choices = [10,100,500])
    parser.add_argument("-s", help = "similarity metric used for knn ", choices = ['dotp','cosine'])
    parser.add_argument("-w", help = "approach for combining prediction given knn", choices = ['mean','weight'])
    args = parser.parse_args()


    output = "-".join([args.m, str(args.k), args.s, args.w])+ '.txt'
    train = 'train.csv'
    dev = 'dev.csv'
    if args.m == 'memory':
        M = readHelper.readTrainMemory(train)
        query, tuples = readHelper.readQueryMemory(dev)
        pred = memory.memoryCF(M, query, args.k, args.s, args.w)
        writeHelper.writePredMemory(output, pred, tuples)
    else:
        M = readHelper.readTrainModel(train)
        tuples = readHelper.readQueryModel(dev)
        pred = model.modelCF(M, tuples, args.k, args.s, args.w)
        writeHelper.writePredModel(output, pred)

    print 'time: %s seconds' % (time.time()-start_time)

if __name__ == '__main__':
    main()