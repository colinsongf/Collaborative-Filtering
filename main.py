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
    parser.add_argument("-p", help = "if standardization used", action='store_true', default = False)
    args = parser.parse_args()
    print args
    output = "-".join([args.m, str(args.k), args.s, args.w])
    if args.p:
        output += '-pcc'
    output += '.txt'
    train = 'train.csv'
    dev = 'dev.csv'
    if args.m == 'memory':
        M = readHelper.readTrainMemory(train)
        query, tuples = readHelper.readQueryMemory(dev)
        if args.p:
            pred = memory.pccMemoryCF(M, query, args.k, args.s, args.w)
        else:
            pred = memory.memoryCF(M, query, args.k, args.s, args.w)
        print min(pred.values())
        print max(pred.values())
        writeHelper.writePredMemory(output, pred, tuples)
    else:
        M = readHelper.readTrainModel(train)
        tuples = readHelper.readQueryModel(dev)
        if args.p:
            pred = model.pccModelCF(M, tuples, args.k, args.s, args.w)
        else:
            pred = model.modelCF(M, tuples, args.k, args.s, args.w)
        print min(pred)
        print max(pred)
        writeHelper.writePredModel(output, pred)

    print 'time: %s seconds' % (time.time()-start_time)

if __name__ == '__main__':
    main()