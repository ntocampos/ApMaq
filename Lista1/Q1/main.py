from random import shuffle
from operator import itemgetter
import math

f = open('breast-cancer-wisconsin.data', 'r')

dataset = []
attr_count = len(f.readline().split(','))
class_index = attr_count - 1

f.seek(0)
for line in f:
    line = line.rstrip().split(',')
    if(len(line) == attr_count and not '?' in line):
        dataset.append(line)

dataset_size = len(dataset)
trainning_size = int(round(len(dataset) * 0.6, 0))
test_size = dataset_size - trainning_size

shuffle(dataset)
trainning_set = dataset[0:trainning_size]
test_set = dataset[trainning_size:]

def kNN(k, x):
    global trainning_set
    global trainning_size
    global class_index
    
    distances = []
    for t in trainning_set:
        distances.append({'id': t[0], 'distance': euclidianDistance(t, x), 'class': t[class_index]})
    
    sorted_list = sorted(distances, key=itemgetter('distance'))
    
    if k == 1:
        return sorted_list[0]['class']
    else:
        knn = sorted_list[0:k]
        
        counter = {}
        for element in knn:
            if element['class'] in counter:
                counter[element['class']] = counter[element['class']] + 1
            else:
                counter[element['class']] = 1
                
        m = 0
        cl = -1
        for i in counter:
            if counter[i] > m:
                m = counter[i]
                cl = i
                
        return cl
        
    
    
def euclidianDistance(x, y):
    s = 0
    
    # starting at 1 to ignore the ID and class, irrelevant to the classification
    for i in range(1, len(x) - 1):
        s = s + ((int(x[i]) - int(y[i])) ** 2)
    
    distance = math.sqrt(s)
    return distance