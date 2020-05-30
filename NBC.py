# -*- coding: utf-8 -*-
import csv
import numpy as np
from math import exp
from math import pi
from math import sqrt
from math import log10
from random import seed
from random import randrange

"""Here we will perform the calculations necessary to use Naive Bayes to determine posterior probabilities."""
            
# Split the dataset based on class values into a dictionary
def separate_by_class(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i] #each instance
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = []
		separated[class_value].append(vector)
	return separated

# Calculate mean, standard deviation and length of a column
def summarize_dataset(dataset):
	summaries = [(np.mean(column), np.std(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1]) #remove class variable statistics
	return summaries

"""We can now find class-wise mean and standard deviation of the columns, as well as the number of rows per class.
This is our actual training step."""
def cl_summarize(dataset):
    summ = {}
    for class_value, rows in dataset.items():
        summ[class_value] = summarize_dataset(rows)
    return summ
    
"""Assuming that the frequencies for each word follow a normal/Gaussian distribution, the following function calculates
the probability that a particular attribute has a particular value. However, in this step we will add a pseudo-count 
of 1 to prevent a situation of dividing by zero, as many means and STDs are still 0."""
def cal_prob(x, mean, stdev):
    e = exp(-(((x-mean)**2 + 1) / ((2 * stdev**2 ) + 1)))
    return (2 / ((sqrt(2 * pi) * stdev)+1)) * e    

"""Now we can use all the above data to calculate the likelihoods of data given a class. Logarithms are used as most of 
the data is extremely small and may vanish. From the likelihoods, the posterior probability of the class can be determined."""
def cal_class_prob(sum_data, inst): #given an instance, returns a list of probabilities for each class
    probabilities = {}
    total = 0 #total number of instances
    for k in sum_data.keys():
        total += len(sum_data[k]) #add number of instances for each class
    for classes, summaries in sum_data.items():
        probabilities[classes] = log10(sum_data[classes][0][2]/float(total)) #log of prior for the class
        for i in range(len(summaries)):
             mean, stdev, count = summaries[i]
             probabilities[classes] += log10(cal_prob(inst[i], mean, stdev)) #add log of likelihood for each attribute
    """The dictionary now contains the Naive Bayes numerator, but it still has to be normalised."""
    s = sum(probabilities.values()) #sum of all probabilities
    for k in probabilities.keys():
        probabilities[k] = probabilities[k]/s #normalising the value
    return probabilities

"""Based on the above function, we can make predictions. The final class will be the one with the most value."""
def predict(summaries, row):
	probabilities = cal_class_prob(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label
    
"""Now that we can perform predictions, the next step is to train the classifier on the data. First, we will do
10-fold cross-validation. The following function will return a list of 10 folds, each containing 400 instances."""
def cross_val(dt):
	dataset_split = []
	dataset_copy = list(dt)
	fold_size = int(len(dt) / 10)
	for _ in range(10):
		fold = []
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy)) 
			fold.append(dataset_copy.pop(index)) #randomly allocate instances to a fold
		dataset_split.append(fold)
	return dataset_split

"""The following metric calculates accuracy."""
def accuracy_metric(actual, predict):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predict[i]: #actual class matches prediction
			correct += 1
	return (correct / len(actual)) * 100.0

"""Now we can evaluate an algorithm using 10-fold cross-validation."""
def eval_algo(dataset, algo):
    folds = cross_val(dataset)
    scores_val = []
    for fold in folds:
        train_set = list(folds) #store all data
        train_set.remove(fold) #remove current fold
        train_x = [] #training set
        for i in train_set:
            for j in i:
                train_x.append(j)
        val_set = [] #validation set
        for row in fold:
            row_copy = list(row)
            val_set.append(row_copy[:-1]) #copy each instance of current fold to validation set
        predicted = algo(train_x, val_set)
        actual = [row[-1] for row in fold] #store actual classes of validation set
        accuracy_val = accuracy_metric(actual, predicted) #find accuracy on validation set
        scores_val.append(accuracy_val)
    return scores_val, np.mean(scores_val)

"""This function ties it all together."""
def naive_bayes(train, test):
	summ = cl_summarize(separate_by_class(train)) #summarize training data by class
	predictions = []
	for row in test:
		output = predict(summ, row) #predicted class of each instance
		predictions.append(output)
	return predictions

"""The predictions returned by the above function are in the form of numbers, and have to be reconverted to letters."""
def class_con(pred, cv): #takes naive_bayes() output list and a dictionary as input
    fin_cl = [] #stores actual string predictions
    for i in pred:
        for k,v in cv.items():
            if i == v: #integer class equals a value in the dictionary
                fin_cl.append(k) #attach actual class
    return fin_cl
    
"""Now we can test it on the training data."""
seed(1)
t = []
cl = {'A': 1, 'B': 2, 'E': 3, 'V': 4}
with open("train.csv") as csv_file:
        absreader = csv.reader(csv_file, delimiter = ',', quotechar = '|')
        for row in absreader:
            x = list(map(int, row[:-1])) #originally the values are strings, so they are converted to int
            x.append(cl[row[-1]]) #appending class as an integer category
            t.append(x)
scoresv, score_meanv = eval_algo(t, naive_bayes)
print('Scores on validation set: %s' % scoresv)
print('Mean Accuracy: %.3f%%' % score_meanv)