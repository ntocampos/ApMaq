from random import shuffle
from operator import itemgetter
import math
import csv

def kNN(k, x):
    global trainning_set
    global class_index
    
    distances = []
    for t in trainning_set:
        distances.append({'distance': euclidianDistance(t, x), 'class': t[class_index]})
    
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

def dataToFloat(input):
    for i in range(0, len(input)):
        # starting at 1 to ignore the ID, stoping before the last to ignore the class
        for j in range(1, len(input[i]) - 1):
            input[i][j] = float(input[i][j])

    return input

def normalizeInput(input):
    for j in range(1, class_index):
        minimum = 100000.0
        maximum = -999999.0
        for i in range(0, len(input)):
            if(input[i][j] > maximum):
                maximum = input[i][j]
            if(input[i][j] < minimum):
                minimum = input[i][j]

        for i in range(0, len(input)):
            input[i][j] = float((input[i][j] - minimum) / (maximum - minimum))

    return input

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

    dataset_size = len(dataset)
    trainning_size = int(round(len(dataset) * 0.7, 0))
    test_size = dataset_size - trainning_size

    dataset = dataToFloat(dataset)
    dataset = normalizeInput(dataset)

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

  
def main():    
    global trainning_set
    global class_index
    k = [1, 2, 3, 5, 7, 9, 11, 13, 15]

    filenames = ['breast-cancer-wisconsin.data', 'wine.data']

    for filename in filenames:
        for i in k:
            runTests(filename, i)

    
if __name__ == "__main__":
    main()

# stem(k, acc); xlabel('K'); ylabel('Accuracy'); title('breast-cancer-wisconsin.data'); xlim([0 15]); ylim([0 1]);