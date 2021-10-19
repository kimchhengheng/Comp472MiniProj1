import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
import warnings
warnings.filterwarnings("ignore")

path = "./BBC/"

data = load_files(path, encoding="latin1")
# data.target_names is the list target name(class name)
classname = data.target_names
# data.data is numpy.ndarray of all the attribute (feature)
# data.target is list is the list of target in numberic the same size with data.data
X , y = data.data, data.target

classnum, counts = np.unique(data.target, return_counts=True)
dataset={}
# create the dictionary of classname and the number of txt within it
for i in classnum:
    dataset[classname[i]] = counts[i]

# plot the distribution of class 
xaxis =[]
xaxislabel =[]
for i in range(len(classname)):
    xaxis.append(i+1)
    xaxislabel.append(classname[i][:5]+": "+str(counts[i]))

barplt = plt.bar(xaxis, counts, tick_label = xaxislabel,
        width = 0.7)
plt.savefig("BBC-distribution.pdf")
plt.show()

# Preprocessing Data
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
# split the data into train set of 80% and test set of 20% which random state is None
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8, random_state=None)
countVect = CountVectorizer()
# using instance of CountVectorizer to fit(X_train) then transform to term-document matrix( in number so machine can understand)
X_train_fitTran = countVect.fit_transform(X_train)


# when fit the data, countVect learn and return the feature back 
# when transform , examine the document-term matrix

# Model selection
from sklearn.naive_bayes import MultinomialNB
multinomialNB1 = MultinomialNB()
# by default MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
# train the multinomialNB
multinomialNB1.fit(X_train_fitTran, y_train)
# before predict tranfrom the test set using the countvectorizer
# during training, we do fit transform
# in model testing we do only transform
y_pred_class = multinomialNB1.predict(countVect.transform(X_test))

# function to write the output to file
def writeTofile(model, numtry ,mode, confmatrix,class_report,prior_prob,n_features,num_token_each,total_token,num_zero_dic,perc_zero_dic,num_zero,per_zero,fave_word,fave_word_log_prob):
    file_name ="./bbc-performance.txt" 
    
    with open(file_name, mode) as textfile:  
        textfile.writelines("(a) "+model+", try "+str(numtry)+"\n")  
        textfile.writelines("\n(b) The confusion matrix\n")
        for line in confmatrix:
            textfile.writelines("\t"+ '\t'.join([str(x) for x in line])+"\n")
        textfile.writelines("\n(c) The precision, recal and F1-measure\n")
        textfile.writelines("\n(d) The accuracy, macro-average F1 and Weigh-average F1\n")
        textfile.writelines(class_report)
        textfile.writelines("\n(e) the prior probability of each class(consider only the test class 80% = 1780)\n")
        for k,v in prior_prob.items():
            textfile.writelines("\t"+k +" has probability : "+str(v)+"\n")
        textfile.writelines("\n(f) the size of the vocabulary ")
        textfile.writelines(str(n_features)+"\n")
        textfile.writelines("\n(g) the number of word-tokens in each class\n")
        for key,val in num_token_each.items():
            textfile.writelines("\t"+key +" has word-tokens : "+str(val)+"\n")
        textfile.writelines("\n(h) the number of word-tokens in the entire corpus ")
        textfile.writelines(str(total_token)+"\n")
        textfile.writelines("\n(i) the number and percentage of words with a frequency of zero in each class\n")
        for key,val in num_zero_dic.items():
            textfile.writelines("\t"+key +" has number of word with a frquency of zero : "+str(val)+"\n")
        textfile.writelines("\n")
        for key,val in perc_zero_dic.items():
            textfile.writelines("\t"+key +" has percentage of word with a frquency of zero : "+str(val)+"\n")
        textfile.writelines("\n(j) the number and percentage of words with a frequency of zero in the entire corpus\n")
        textfile.writelines("\tThe number of word with a frquency of zero within the corpus " + str(num_zero)+"\n")
        textfile.writelines("\tThe number of word with a frquency of zero within the corpus " + str(per_zero)+"\n")
        textfile.writelines("\n(k) your 2 favorite words (that are present in the vocabulary) and their log-prob\n")
        for j in range(len(fave_word_log_prob)):
            textfile.writelines("\tword "+fave_word[j] + " has the log probbility of "+str(fave_word_log_prob[j])+"\n")
        textfile.writelines("\n")
    textfile.close()

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def extractValueWriteToFile(model,modelname, numtry ,mode ):
    confmatrix = confusion_matrix(y_test, y_pred_class)
    class_report =classification_report(y_test, y_pred_class)
    prop = [round(c/2225,4) for c in model.class_count_]
    prior_prob = dict(zip(classname,prop))
    n_features = model.n_features_in_ # number of feature, the same len(countVect.get_feature_names())
    feature_count = model.feature_count_ # array of frequency, sum all value of each class, one row represend 1 class 
    sum_feature_each =  np.sum(feature_count, axis = 1)
    num_token_each = dict(zip(classname,sum_feature_each))
    total_token = np.sum(sum_feature_each)
    num_zero_eachClass =[(np.array(line).size-np.count_nonzero(line)) for line in feature_count]
    percen_zero_eachClass =[]
    for i in range(len(num_token_each)):
        percen_zero_eachClass.append(round(num_zero_eachClass[i]/sum_feature_each[i],4))
    num_zero_dic = dict(zip(classname,num_zero_eachClass))
    perc_zero_dic =  dict(zip(classname,percen_zero_eachClass))
    num_zero =np.array(feature_count).size - np.count_nonzero(feature_count)
    per_zero = round(num_zero/total_token,4)
    fave_word = [countVect.get_feature_names()[-10],countVect.get_feature_names()[-100]]
    fave_word_log_prob = [model.feature_log_prob_[:, -10],model.feature_log_prob_[:, -100]]

    writeTofile(modelname, numtry ,mode ,confmatrix,class_report, prior_prob, n_features,num_token_each,total_token,num_zero_dic,perc_zero_dic,num_zero,per_zero,fave_word,fave_word_log_prob)

extractValueWriteToFile(multinomialNB1,"MulitnomialNB default values",1,'w')

# rerun again the value does not change it suppose to change the number of feature
extractValueWriteToFile(multinomialNB1,"MulitnomialNB default values",2,'a')

#  smoothing1
multinomialNBSm1 = MultinomialNB(alpha=0.0001)
multinomialNBSm1.fit(X_train_fitTran, y_train)
y_pred_class = multinomialNBSm1.predict(countVect.transform(X_test))
extractValueWriteToFile(multinomialNBSm1,"MulitnomialNB with smoothing value to 0.0001",3,'a')

#  smoothing 2
multinomialNBSm2 = MultinomialNB(alpha=0.0009)
multinomialNBSm2.fit(X_train_fitTran, y_train)
y_pred_class = multinomialNBSm2.predict(countVect.transform(X_test))
extractValueWriteToFile(multinomialNBSm2,"MulitnomialNB with smoothing value to 0.9",4,'a')

# print("finish")
