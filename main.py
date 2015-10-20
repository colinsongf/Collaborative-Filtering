__author__ = 'Ariel'


import numpy as np
import time
import readHelper
import KNN

start_time = time.time()

M = readHelper.readTrain('train.csv')

query = readHelper.readQuery("dev.csv")


