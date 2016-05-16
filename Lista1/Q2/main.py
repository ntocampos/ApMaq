from random import shuffle
from operator import itemgetter
import math
import csv

class Attribute:
    classes = [{}]
    total = 0

def kNN(k, x):
    global trainning_set
    global class_index
    
    distances = []
    for t in trainning_set:
        distances.append({'distance': vdmDistance(t, x, 2), 'class': t[class_index]})
    
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
    
def countClasses():
    classes = []
    for data in trainning_set:
        if(data[class_index] not in classes):
            classes.append(data[class_index])

    return classes

def countAttr():
    global trainning_set
    attr = []
    for i in range(class_index):
        attr.append({})

    for data in trainning_set:
        for i in range(0, class_index):
            if data[i] not in attr[i]:
                attr[i][data[i]] =  1
            else:
                attr[i][data[i]] += 1

    return attr

def countAttr2(attr, i):
    global trainning_set
    
    return float(len([x for x in trainning_set if x[i] == attr]))

def countAttrClasses(attr, i, c):
    global trainning_set
    
    return float(len([x for x in trainning_set if x[i] == attr and x[class_index] == c]))

def Niac(i, val, cl):
    global trainning_set
    count  = 0

    for data in trainning_set:
        if(data[i] == val and data[class_index] == cl):
            count += 1

    return float(count)
 
def vdmDistance(x, y, q = 2):
    classes = countClasses()
    attr = countAttr()
    sum_attr = 0.0

    for i in range(0, class_index):
        sum_cl = 0.0
        for c in classes:
            #sum_cl += abs(countAttrClasses(x[i], i, c) / attr[i][x[i]] - countAttrClasses(y[i], i, c) / attr[i][y[i]])**q
            #print str(countAttrClasses(x[i], i, c)) +  "/" + str(countAttr2(x[i], i)) + "-" + str(countAttrClasses(y[i], i, c)) + "/" + str(countAttr2(y[i], i))
            sum_cl += abs(countAttrClasses(x[i], i, c) / countAttr2(x[i], i) - countAttrClasses(y[i], i, c) / countAttr2(y[i], i))**q

        sum_attr += sum_cl

    return math.sqrt(sum_attr)

def preprocessData(filename):
    global class_index

    f = open(filename, 'r')

    dataset = []
    attr_count = len(f.readline().split(','))
    class_index = attr_count - 1

    f.seek(0)
    for line in f:
        line = line.rstrip().split(',')
        if(len(line) == attr_count and not '?' in line):
            dataset.append(line)

    dataset = dataset[1:]

    dataset_size = len(dataset)
    trainning_size = int(round(len(dataset) * 0.3, 0))
    test_size = dataset_size - trainning_size

    shuffle(dataset)
    
    trainning_set = dataset[0:trainning_size]
    test_set = dataset[trainning_size:]

    return trainning_set, test_set

def runTests(filename, k):
    global trainning_set
    trainning_set, test_set = preprocessData(filename)

    i = 0;
    for t in test_set:
        c = kNN(k, t)
        if(c != t[class_index]):
            # print "Classification: " + c + " Correct answer: " + t[class_index]
            i = i + 1

    print "######## K = " + str(k) + " ########"
    print "Dataset size: " + str(len(trainning_set) + len(test_set))
    print "Trainning set size: " + str(len(trainning_set))
    print "Test set size: " + str(len(test_set))
    print "Errors: " + str(i)
    print "Accuracy: " + str(1 - float(i) / float(len(test_set)))
    print "\n"

    return 1 - float(i) / float(len(test_set))

  
def main():    
    global trainning_set
    global class_index
    k = [1, 2, 3, 5, 7, 9, 11, 13, 15]
    acc = []

    filenames = ['diagnosis2.data', 'pro-bloggers.data']

    for filename in filenames:
        acc.append(filename)
        for i in k:
            acc.append(runTests(filename, i))

    print k
    print acc
    
if __name__ == "__main__":
    main()

# stem(k, acc); xlabel('K'); ylabel('Accuracy'); title('breast-cancer-wisconsin.data'); xlim([0 15]); ylim([0 1]);