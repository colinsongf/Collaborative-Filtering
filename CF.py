__author__ = 'Ariel'

import numpy as np
import KNN


def memoryCF(m, query, k, func, func_com):
    res = {}
    for user in sorted(query.keys()):
        knn_idx, knn_value = KNN.knn(m, m[user], k, func)
        prediction = KNN.combine(m, knn_idx, knn_value, func_com)
        for movie in query[user]:
            res[user, movie] = prediction[movie]
    return res