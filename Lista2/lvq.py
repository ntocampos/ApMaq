# LVQ1
from random import shuffle
from operator import itemgetter
import math
import numpy as np

# Converts the numerical data to float, except the class and ID
# Returns a list of data instances with attributes converted to float
def dataToFloat(input):
    for i in range(0, len(input)):
        # starting at 1 to ignore the ID, stoping before the last to ignore the class
        for j in range(1, len(input[i]) - 1):
            input[i][j] = float(input[i][j])

    return input


# Normalize the data set numerical values between [0 1] interval
# Returns a list of data instances with normalized attributes
def normalizeInput(data):
    for j in range(1, class_index):
        minimum = 100000.0
        maximum = -999999.0
        for i in range(0, len(data)):
            if(data[i][j] > maximum):
                maximum = data[i][j]
            if(data[i][j] < minimum):
                minimum = data[i][j]

        for i in range(0, len(data)):
            data[i][j] = float(data[i][j] - minimum) / (maximum - minimum)

    return data


# Reads the dataset from file
# Returns a list of raw data instances
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
    prototypes_size = int(dataset_size * 0.4) # 20 per cent to prototypes
    trainning_size = int(dataset_size * 0.4) # 50 percent to trainning
    test_size = dataset_size - trainning_size - prototypes_size # The remaining to test (apprx. 30 per cent)

    dataset = dataToFloat(dataset)
    dataset = normalizeInput(dataset)

    shuffle(dataset)
    
    prototypes_set = dataset[0:prototypes_size]
    trainning_set = dataset[prototypes_size:prototypes_size + trainning_size]
    test_set = dataset[prototypes_size + trainning_size:]

    return prototypes_set, trainning_set, test_set


# Calculate Euclidian distance
# Returns float number
def euclidianDistance(x, y):
    s = 0
    
    # starting at 1 to ignore the ID and class, irrelevant to the classification
    for i in range(1, len(x) - 1):
        s = s + ((x[i] - y[i]) ** 2)
    
    distance = math.sqrt(s)
    return distance


# Finds the class of the k nearest neighbors (by frequency when n > 1)
# Returns a string containing the class
def kNN(k, x, trainning_set):
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

# Finds the closest prototype to x instance
# Returns the float distance between them and the prototype itself
def findClosestNeighbor(x, prototypes_set, n):
    global class_index

    distances = []
    for i in range(0, len(prototypes_set)):
    	p = prototypes_set[i]
    	distances.append({'i': i, 'distance': euclidianDistance(p, x), 'prototype': p})

    sorted_list = sorted(distances, key=itemgetter('distance'))

    if n == 2:
        return sorted_list[0], sorted_list[1]
    else:
        return sorted_list[0]

# Get the parameter 'proto' closer to x
# Returns the new prototype
def getProtoCloser(x, proto, alpha):
	global class_index

	new_proto = [0] * len(proto)
	new_proto[0] = proto[0]
	new_proto[class_index] = proto[class_index]

	# Ignore ID and class (first and last element)
	for i in range(1, class_index):
		# w(t+1) = w(t) + a*(x - w(t))
		new_proto[i] = proto[i] + alpha * (x[i] - proto[i])

	return new_proto

# Get the parameter 'proto' further to x
# Returns the new prototype
def getProtoFurther(x, proto, alpha):
	global class_index

	new_proto = [0] * len(proto)
	new_proto[0] = proto[0]
	new_proto[class_index] = proto[class_index]

	# Ignore ID and class (first and last element)
	for i in range(1, class_index):
		# w(t+1) = w(t) - a*(x - w(t))
		new_proto[i] = proto[i] - alpha * (x[i] - proto[i])

	return new_proto


def LVQ1(filename):
    global closer, further
    closer = further = 0
    # 1) Select some prototypes (let's do it randomly)
    prototypes_set, trainning_set, test_set = preprocessData(filename)
    prototypes_set_original = list(prototypes_set)
    iterations = 20 # Number of adjustment iterations
    # Array of alphas
    alphas_array = list(reversed(np.linspace(0.1, 0.9, iterations).tolist()))

    for k in range(iterations):
        # 2) For every case x in dataset (trainning_set):
        for x in trainning_set:
            # 2.1) Find the closest prototype p
            closest = findClosestNeighbor(x, prototypes_set, 1)
            i = closest['i']
            distance = closest['distance']
            proto = closest['prototype']

            # 2.2) If class(p) == class(x) then get them closer
            if x[class_index] == proto[class_index]:
                new_proto = getProtoCloser(x, proto, alphas_array[k])
                closer = closer + 1

            # 2.3) If not, get them more distant
            else:
                new_proto = getProtoFurther(x, proto, alphas_array[k])
                further = further + 1

            prototypes_set[i] = new_proto

        iterations = iterations + 1

    return prototypes_set, prototypes_set_original, trainning_set, test_set

def LVQ21(filename):
    # 1) Select some prototypes (using LVQ as selection)
    prototypes_set, prototypes_set_original, trainning_set, test_set = LVQ1(filename)

    e = float(len(trainning_set)) / (len(prototypes_set) + len(trainning_set) + len(test_set)) # Epsilon for the window
    iterations = 20 # Number of adjustment iterations
    # Array of alphas
    alphas_array = list(reversed(np.linspace(0.1, 0.9, iterations).tolist()))

    for k in range(iterations):
        # 2) For every case x in dataset (trainning_set):
        for x in trainning_set:
            # 2.1) Find the closest prototype p
            c1, c2 = findClosestNeighbor(x, prototypes_set, 2)
            i1 = c1['i']
            distance1 = c1['distance']
            proto1 = c1['prototype']

            i2 = c2['i']
            distance2 = c2['distance']
            proto2 = c2['prototype']
            
            if distance1 == 0:
                distance1 = np.finfo(float).eps
            if distance2 == 0:
                distance2 = np.finfo(float).eps

            # 2.2) If class(p) == class(x) then get them closer
            if proto1[class_index] != proto2[class_index]:
                d12 = distance1/distance2
                d21 = distance2/distance1
                if min(d12, d21) > (1 - e) and max(d12, d21) < (1 + e):
                    if (x[class_index] == proto1[class_index]):
                        new_proto1 = getProtoCloser(x, proto1, alphas_array[k])
                        new_proto2 = getProtoFurther(x, proto2, alphas_array[k])
                        prototypes_set[i1] = new_proto1
                        prototypes_set[i2] = new_proto2

                    elif (x[class_index] == proto2[class_index]):
                        new_proto1 = getProtoFurther(x, proto1, alphas_array[k])
                        new_proto2 = getProtoCloser(x, proto2, alphas_array[k])
                        prototypes_set[i1] = new_proto1
                        prototypes_set[i2] = new_proto2

        iterations = iterations + 1


    return prototypes_set, prototypes_set_original, trainning_set, test_set

def LVQ3(filename):
    # 1) Select some prototypes (using LVQ2.1 as selection)
    prototypes_set, prototypes_set_original, trainning_set, test_set = LVQ21(filename)

    e = float(len(trainning_set)) / (len(prototypes_set) + len(trainning_set) + len(test_set)) # Epsilon for the window
    beta = 0.2
    iterations = 20 # Number of adjustment iterations
    # Array of alphas
    alphas_array = list(reversed(np.linspace(0.1, 0.9, iterations).tolist()))

    for k in range(iterations):
        # 2) For every case x in dataset (trainning_set):
        for x in trainning_set:
            # 2.1) Find the closest prototype p
            c1, c2 = findClosestNeighbor(x, prototypes_set, 2)
            i1 = c1['i']
            distance1 = c1['distance']
            proto1 = c1['prototype']

            i2 = c2['i']
            distance2 = c2['distance']
            proto2 = c2['prototype']
            
            if distance1 == 0:
                distance1 = np.finfo(float).eps
            if distance2 == 0:
                distance2 = np.finfo(float).eps

            # 2.2) If class(p) == class(x) then get them closer
            if proto1[class_index] != proto2[class_index]:
                d12 = distance1/distance2
                d21 = distance2/distance1
                if min(d12, d21) > (1 - e) and max(d12, d21) < (1 + e):
                    if (x[class_index] == proto1[class_index]):
                        new_proto1 = getProtoCloser(x, proto1, alphas_array[k])
                        new_proto2 = getProtoFurther(x, proto2, alphas_array[k])
                        prototypes_set[i1] = new_proto1
                        prototypes_set[i2] = new_proto2

                    elif (x[class_index] == proto2[class_index]):
                        new_proto1 = getProtoFurther(x, proto1, alphas_array[k])
                        new_proto2 = getProtoCloser(x, proto2, alphas_array[k])
                        prototypes_set[i1] = new_proto1
                        prototypes_set[i2] = new_proto2
            elif x[class_index] == proto1[class_index]:
                # This means that the two closest prototypes are from the same class of x
                new_proto1 = getProtoCloser(x, proto1, beta * alphas_array[k])
                new_proto2 = getProtoCloser(x, proto2, beta * alphas_array[k])
                prototypes_set[i1] = new_proto1
                prototypes_set[i2] = new_proto2

        iterations = iterations + 1


    return prototypes_set, prototypes_set_original, trainning_set, test_set

def runTests(trainning_set, test_set, k):
    i = 0;
    for t in test_set:
        c = kNN(k, t, trainning_set)
        if(c != t[class_index]):
            print("Classification: " + str(c) + " Correct answer: " + str(t[class_index]))
            i = i + 1

    print("######## K = " + str(k) + " ########")
    print("Dataset size: " + str(len(trainning_set) + len(test_set)))
    print("Trainning set size: " + str(len(trainning_set)))
    print("Test set size: " + str(len(test_set)))
    print("Errors: " + str(i))
    print("Accuracy: " + str(1 - float(i) / len(test_set)))
    print("\n")


def main():
    global class_index
    #k = [1, 3]

    filenames = ['breast-cancer-wisconsin.data']

    for filename in filenames:
        proto, proto_original, trainning_set, test = LVQ3(filename)
        print("# Test with prototypes #")
        runTests(proto, test, 3)

        print("# Test with original prototypes #")
        runTests(proto_original, test, 3)

    
if __name__ == "__main__":
    main()

# stem(k, acc); xlabel('K'); ylabel('Accuracy'); title('breast-cancer-wisconsin.data'); xlim([0 15]); ylim([0 1]);