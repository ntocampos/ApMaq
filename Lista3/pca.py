import numpy as np
import matplotlib.pyplot as plt
import knn

# Runs the principal component analysis
# [in] data: dataset already shuffled with classes in the last row
# [in] n_components: desired number of components for the new dataset
# [out] new_dataset: the reduced dataset with 'n_components' columns
def PCA(dataset, n_components):
	means = np.mean(dataset, axis = 0)

	normalized = np.zeros([len(dataset), len(dataset[0])])
	for i in range(len(dataset)):
		for j in range(len(dataset[i])):
			normalized[i][j] = dataset[i][j] - means[j]

	# Estimate the covariance matrix with the columns being
	# the variables (rowvar arg)
	c_matrix = np.cov(normalized, rowvar = 0)

	# Calculate the eigenvalues and eigenvectors of the covariance
	# matrix
	vals, vecs = np.linalg.eig(c_matrix)

	# Create a list of tuples with the first argument being the 
	# eigenvalue and the second being the respective eigenvector
	pairs = [(np.abs(vals[i]), vecs[:, i]) for i in range(len(vals))]

	# Then, do a descending sort
	pairs.sort()
	pairs.reverse()

	# Get the 'n_components' more significant eingenvectors to compose the transformation
	# matrix 'w'
	w = np.hstack(([pairs[i][1].reshape(len(vals), 1) for i in range(n_components)]))

	# Now, do the dot multiplication of the normalized input matrix with w
	new_dataset = normalized.dot(w)

	return new_dataset

# Separates dataset in two sets: trainning and test
def getSets(dataset, classes):
	l = len(dataset)	

	trainning = 0.5
	return dataset[0:round(l * trainning)], classes[0:round(l * trainning)], dataset[round(l * trainning):], classes[round(l * trainning):]

def runTests(trainning, tr_classes, test, t_classes, k):
	errors = 0;
	for i in range(len(test)):
		c = knn.kNN(k, test[i], trainning, tr_classes)
		if(c != t_classes[i]):
			#print("Classification: " + str(c) + " Correct answer: " + str(t_classes[i]))
			errors = errors + 1

	print("######## K = " + str(k) + " ########")
	print("Dataset size: " + str(len(trainning) + len(test)))
	print("Trainning set size: " + str(len(trainning)))
	print("Test set size: " + str(len(test)))
	print("Errors: " + str(errors))
	print("Accuracy: " + str(1 - float(errors) / len(test)))
	print("\n")

	return (1 - errors / len(test))


data = np.loadtxt('glass.data', delimiter = ',')
data = np.delete(data, 0, 1) # Delete the ID column
np.random.shuffle(data)

dataset = data[:, 0:len(data[0]) - 1]
classes = data[:, len(data[0]) - 1]

trainning, tr_classes, test, t_classes = getSets(dataset, classes)

n_components = range(1, len(dataset[0]) + 1)
acc = []
for n in n_components:
	pca_data = PCA(dataset, n)
	trainning_pca, tr_pca_classes, test_pca, t_pca_classes = getSets(pca_data, classes)

	acc.append(runTests(trainning_pca, tr_pca_classes, test_pca, t_pca_classes, 3))

plt.plot(n_components, acc, 'ro')
plt.axis([0, len(dataset[0]) + 1, 0, 1])
plt.xticks(range(0, len(dataset[0]) + 1, 1))
plt.xlabel('Number of components (PCA)')
plt.ylabel('Accuracy')
plt.title('Components x Accuracy')
plt.grid(True)
plt.show()