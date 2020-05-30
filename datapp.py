# -*- coding: utf-8 -*-
"""The aim of this code is to convert the given trg.csv or tst.csv files into a dataset that can be used by 
Naive Bayes. This will be done by extracting the non-common words of the abstracts and calculating their frequencies.
Initially everything will be stored in dictionaries, but it will finally be converted into a list of lists
and stored in another csv file for use by the next code."""
import csv
import numpy as np

def check_com(w):
    """Some words are very common. Their frequencies are irrelevant for us. This function checks from a
    list of common words to find out if a word is common or not."""
    comw = ["the", "of", "and",	"a", "to", "in", "is", "that", "it", "was", "for", "on", "are", "as", "with", "they", 
        "be", "at", "this", "have", "from", "or", "one", "had", "by", "word", "but", "not", "what", "all", "were", 
        "we", "when", "can", "there", "use", "an", "each", "which", "into", "time", "has", "look", "do", "how", 
        "their", "if", "will", "up", "other", "about", "out", "many", "then", "them", "these", "so", "some", "would",
        "make", "like", "call", "two", "more", "write", "go", "see", "number", "no", "way", "could", "than", "first",
        "water", "been", "who", "oil", "its", "now", "find", "long", "down", "did", "get", "come", "made", "may", "part"]
    if w not in comw:
        return True
    else:
        return False
    
words = [] #list storing all non-common words encountered in training set
with open('trg.csv') as csv_file:
    wreader = csv.reader(csv_file, delimiter = ',', quotechar = '|')
    for row in wreader:
        if (row[0] != 'id'): #ignoring first line
            r = row[2].split(' ') #splitting the abstract
            if(len(r[0][1:]) > 3 and check_com(r[0][1:])): #
                """The string is spliced as first word in each case consists of a quotation mark. Only words larger
                than 3 letters are considered, otherwise the dataset will become massive."""
                words.append(r[0][1:])
            for w in r:
                if len(w) > 3 and check_com(w): 
                    words.append(w) #store non-common words in words
            if(len(r[-1]) > 3 and check_com(r[-1][:-1])): #the last word of each abstract also has an ending quotation mark
                words.append(r[-1][:-1])
words = list(np.unique(np.sort(words))) #removing duplicates

"""The following function preprocesses the data stored in file x. Note that the .py file should be running in the 
same directory as x for it to work. The function returns a list of lists containing all the attributes and class
for each instance in the training or test set."""
def prep(x): 
    t2 = [] #list of 4000 dictionaries
    with open(x) as csv_file:
        absreader = csv.reader(csv_file, delimiter = ',', quotechar = '|')
        for row in absreader:
                if (row[0] != "id"): #ignoring first line of the csv file
                    t1 = {} #stores word frequencies for non-common words in each abstract, with the class of that abstract
                    for i in words:
                        t1[i] = 0
                    r = row[-1].split(' ') #r has the abstract 
                    fw = r[0][1:] #first word
                    if(fw in words):
                        """words not encountered in the training set won't be considered"""
                        t1[fw] += 1 #increment the frequency of the encountered word
                    l = len(r) - 1
                    for i in range(1, l-1):
                        if (r[i] in words):
                            t1[r[i]] += 1
                    lw = r[l-1][:-1] #last word
                    if (lw in words):
                        t1[lw] += 1
                    if(len(row) == 3): #this checks if x contains the training data, where row would have 3 items
                        t1['zz'] = row[1]
                        """The above command adds the class to t1. The reason it is represented by 'zz' is because
                        'class' itself may be a word in the dictionary and if so, it would be overwritten by the 
                        classification."""
                    t2.append(t1) #store the instance in t2    
    """Now that we have our data as a list of dictionaries, the next step is to store it in a CSV format.
    This is done to prevent the huge runtimes required to do the above steps in the future. To store the 
    final training data for later use, we will first convert it into a list of lists by removing the labels."""
    dt = []
    for i in t2:
        x = list(i.values())
        dt.append(x)
    return dt

# Calculate mean and standard deviation of the data
def summarize_dataset(dataset):
	summaries = [(np.mean(column), np.std(column)) for column in zip(*dataset)]
	return summaries

"""The prep() function gives us a list of lists. This can be stored in a CSV file now.
However, we will do another step of preprocessing. A lot of columns have zero mean and standard deviation.
They can be discarded from both training and test data."""
trdt = prep('trg.csv') #acquire training data
tsdt = prep('tst.csv') #Acquire test data
strdt = []
for i in trdt:
    strdt.append(list(map(int, i[:-1]))) #adding list of integer attributes for each instance
strdt = summarize_dataset(strdt)
off = 0 
"""When columns are deleted, other columns shift to the left. For instance, if column 0 
were deleted, the other columns would move left. After this, if column 2 were to be deleted,
then using del function with 2 as a parameter would actually delete column 3, since it now 
occupies the position of 2. The offset accounts for this. It ensures that the wrong column isn't deleted."""
for i in range(len(strdt)):
    if (strdt[i][0] == 0.0 and strdt[i][1] == 0.0): #both mean and STD are zero
        for j in range(4000):
            del trdt[j][i-off] #remove the value for ith column for all instances in training data
        for j in range(1000):
            del tsdt[j][i-off] #remove the value for ith column for all instances in test data
        off += 1

"""Even now, we have some columns with exactly the same value as their neighbouring columns. For example,
'homo' usually appears with 'sapiens'. So columns 'homo' and 'sapiens' would have exactly the same values. 
These duplicates can be removed."""
cols = list(zip(*trdt)) #each element in this list is a list of column values
i = 0
while i < len(trdt[0]): #iterate through each column, the loop runs for the last time on the second last column
    c = 0
    for j,k in zip(cols[i],cols[i+1]): #compare values of adjacent columns
        if j == k:
            c+=1
    if c == len(cols[i]): #all values are equal
        for j in range(4000):
            del trdt[j][i+1] #remove all the values in the adjacent column
        for j in range(1000):
            del tsdt[j][i+1] #remove the corresponding column in test data
    i+=1
        
"""The datasets are much smaller, and can now be stored in CSV."""
with open('train.csv', 'w', newline='') as csvfile:
    abswriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in trdt:
        abswriter.writerow(i)
with open('test.csv', 'w', newline='') as csvfile:
    abswriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in tsdt:
        abswriter.writerow(i)
        
"""Finally, train.csv and test.csv contain the preprocessed data ready to be supplied to NBC."""