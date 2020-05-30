# -*- coding: utf-8 -*-
import NBC #as this is an extension, some classes will be reused from NBC
import csv
from random import seed
from math import log10

"""An extension to the standard Naive Bayes can be the multinomial Naive Bayes classifier. 
We will treat the frequencies as multinomial distributions. First, we will generate the likelihood of all words."""
def lik(data):
    liks = {} #stores all the likelihoods by class
    for k in data:
        tw = 0 #total words in class k
        for r in data[k]: #each r corresponds to an instance
            tw += sum(r) #adds up all the words present in an instance
        """Now we have the total number of words in class k."""
        p_list = [] #list of probabilities per class
        for i in zip(*data[k]): #i corresponds to a column of the dataset, i.e., a word in the dataset
            s = sum(i) #sums up to all occurences of a word in the class k
            """Now that we have the occurences of word i, we can calculate p(k,i). The pseudocount here is 1 for
            the numerator and the total number of unique words,i.e.,total number of columns in the dataset for the denominator."""
            p = (s + 1)/(tw + len(data[k][0]))
            p_list.append(p)
        liks[k] = p_list
    return liks

def p(data,k,i): #given a class and a word(indicated by column number), find its likelihood within that class
    return data[k][i]

"""Now we can calculate the posterior probability, which is directly proportional to the sum of the log of prior probability
and the product of frequency of each word and the log of likelihood of that word in the instance."""
def cal_p(liks, priors, inst): #calculate class probabilities of the given instance
    prob = {}
    for k in priors: #k is the class
        l = log10(priors[k]) #storing the prior of the current class
        for i in range(len(inst)):
            if inst[i] > 0: #checking if a word exists in the instance
                l += inst[i] * log10(p(liks, k, i)) #adding the log of likelihood
        prob[k] = l #store the posterior probability for class k
    """The dictionary now contains the Naive Bayes numerator, but it still has to be normalised."""
    s = sum(prob.values()) #sum of all probabilities
    for k in prob.keys():
        prob[k] = prob[k]/s #normalising the value
    return prob

"""Based on the above function, we can make predictions. The final class will be the one with the most value."""
def predict(probabilities):
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

"""This function ties it all together."""
def multi(train, test):
    data = NBC.separate_by_class(train) #get class-wise instances
    liks = lik(data) #train the classifier by calculating classes of all the data
    predictions = []
    priors = {}
    """The prior probability of a class is the total number of instances in the training set that have that class,
    divided by all instances in the set."""
    for k in range(1,5): #the classes are 1,2,3,4
        priors[k] = len(data[k])/len(train)
    for row in test:
        prob = cal_p(liks, priors, row)
        output = predict(prob) #predicted class of each instance
        predictions.append(output)
    return predictions

"""We can now train and test the multinomial Naive Bayes."""
seed(1)
t = []
cl = {'A': 1, 'B': 2, 'E': 3, 'V': 4}
with open("train.csv") as csv_file:
        absreader = csv.reader(csv_file, delimiter = ',', quotechar = '|')
        for row in absreader:
            x = list(map(int, row[:-1])) #originally the values are strings, so they are converted to int
            x.append(cl[row[-1]]) #appending class as an integer category
            t.append(x)
scoresv, score_meanv = NBC.eval_algo(t, multi)
print('Scores on validation set: %s' % scoresv)
print('Mean Accuracy: %.3f%%' % score_meanv)

"""For the test data, we need to be able to predict the classes."""
with open('test.csv') as csv_file:
    test = [] #stores all test cases
    absreader = csv.reader(csv_file, delimiter = ',', quotechar = '|')
    for row in absreader:
        x = list(map(int, row)) #originally the values are strings, so they are converted to int
        test.append(x)
pr = multi(t,test) #get integer predictions
p_act = NBC.class_con(p, cl) #get string predictions

"""Now we have to store them in a csv file."""
with open('sjha286.csv', 'w', newline='') as csvfile:
    abswriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    abswriter.writerow(['id','class'])
    for i in range(len(p_act)):
        abswriter.writerow([i+1,p_act[i]])