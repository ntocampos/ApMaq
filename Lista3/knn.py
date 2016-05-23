import math
import collections
import numpy as np
from operator import itemgetter

def euclidianDistance(x, y):
    s = 0
    
    # starting at 1 to ignore the ID and class, irrelevant to the classification
    for i in range(len(x)):
        s = s + ((x[i] - y[i]) ** 2)
    
    distance = math.sqrt(s)
    return distance


# Finds the class of the k nearest neighbors (by frequency when n > 1)
# Returns a string containing the class
def kNN(k, x, trainning_set, tr_classes):
    distances = [(euclidianDistance(x, trainning_set[i]), trainning_set[i], tr_classes[i]) for i in range(len(trainning_set))]
    distances = sorted(distances, key = itemgetter(0))

    if k == 1:
        return distances[0][2]
    else:
        distances = distances[0:k]
        c = [x[2] for x in distances]
        counter = collections.Counter(c)
        most_common = counter.popitem()
        return float(most_common[0])