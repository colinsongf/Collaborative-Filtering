__author__ = 'Ariel'

import time
import argparse
import readHelper
import writeHelper
import model
import memory
import BiCluster
import pickle

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description = "collaborative filtering")
    parser.add_argument("-m", help = "memory-based or model collaborative filtering", choices = ['memory','model'])
    parser.add_argument("-k", help = "number of k nearest neighborhood", type = int, choices = [10,100,500])
    parser.add_argument("-s", help = "similarity metric used for knn ", choices = ['dotp','cosine'])
    parser.add_argument("-w", help = "approach for combining prediction given knn", choices = ['mean','weight'])
    parser.add_argument("-p", help = "if standardization used", action = 'store_true', default = False)

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

        writeHelper.writePredMemory(output, pred, tuples)
    else:
        M = readHelper.readTrainModel(train)
        tuples = readHelper.readQueryModel(dev)
        if args.p:
            pred = model.pccModelCF(M, tuples, args.k, args.s, args.w)
        else:
            pred = model.modelCF(M, tuples, args.k, args.s, args.w)

        writeHelper.writePredModel(output, pred)

    print 'time: %s seconds' % (time.time()-start_time)

def getBiCluster(k1, k2):
    train = 'train.csv'
    M = readHelper.readTrainModel(train)
    start_time = time.time()
    u2U, m2M, u2Mcluster, m2Ucluster = BiCluster.BiClustering(M, k1, k2)
    ClusterTime = (time.time()-start_time)
    print 'time: %s seconds' % ClusterTime
    # write to file
    with open('u2U.txt', 'w') as f1:
        pickle.dump(u2U, f1)
    with open('w2W.txt', 'w') as f2:
        pickle.dump(m2M, f2)
    with open('u2Mcluster.txt', 'w') as f3:
        pickle.dump(u2Mcluster, f3)
    with open('m2Ucluster.txt', 'w') as f4:
        pickle.dump(m2Ucluster, f4)
    print 'time: %s seconds' % (time.time()-start_time)
    print ClusterTime
    return ClusterTime
if __name__ == '__main__':
    #main()
    getBiCluster(300,100)
