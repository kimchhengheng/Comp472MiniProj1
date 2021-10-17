from numpy.random import multinomial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

dir_path = "./drug200.csv"
drugdata =pd.read_csv(dir_path)

f1str =  'f1-score' 
acustr = 'accuracy'
macrostr = 'f1_macro'
weistr = 'f1_weighted'
allreport = {
    "NB": {'accuracy':[],'f1_macro':[],'f1_weighted':[]},
    "Base-DT": {'accuracy':[],'f1_macro':[],'f1_weighted':[]},
    "Top-DT": {'accuracy':[],'f1_macro':[],'f1_weighted':[]},
    "PER": {'accuracy':[],'f1_macro':[],'f1_weighted':[]},
    "Base-MLP": {'accuracy':[],'f1_macro':[],'f1_weighted':[]},
    "Top-MLP": {'accuracy':[],'f1_macro':[],'f1_weighted':[]}
}

# print(drugdata.info())
# 200 row , 6 columns 5 feature last one is target

# convert the sex column to number, could use get_dummies , map, or catergory since it is binary 
drugdata['Sex'] = pd.get_dummies(drugdata['Sex'],drop_first=True)

# convert the cholesterol column to category then get code number
drugdata['Cholesterol']= pd.Categorical(drugdata['Cholesterol'],['NORMAL','HIGH'],ordered=True)
drugdata['Cholesterol']= drugdata['Cholesterol'].cat.codes

# convert the BP column to category then get code number
drugdata['BP'] = pd.Categorical(drugdata['BP'],['LOW','NORMAL','HIGH'],ordered=True)
drugdata['BP']= drugdata['BP'].cat.codes

# plot the distribution
classname, counts = np.unique(drugdata['Drug'], return_counts=True)

# plot the distribution of class 
xaxis =[]
xaxislabel =[]
for i in range(len(classname)):
    xaxis.append(i+1)
    xaxislabel.append(classname[i][:5]+": "+str(counts[i]))

barplt = plt.bar(xaxis, counts, tick_label = xaxislabel,
        width = 0.7)
plt.savefig("drug-distribution.pdf")
plt.show()

from sklearn.model_selection import train_test_split

X = drugdata.iloc[:,:-1]
y = drugdata.iloc[:,[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y)

# preprocessing data, encoder the target to numberical value
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_encoder = le.fit_transform(y_train)


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

multinomialNB = MultinomialNB()
multinomialNB.fit(X_train, y_train_encoder)

decisionTreeBase = DecisionTreeClassifier()
decisionTreeBase.fit(X_train, y_train_encoder)

decisionTreeTop = DecisionTreeClassifier()
gd_dctree = [
    { 'max_depth': [2], 'criterion':['entropy'], 'min_samples_split':[3]}
]
gd_dctree=GridSearchCV(decisionTreeTop, param_grid=gd_dctree)
gd_dctree.fit(X_train, y_train_encoder)

perceptron = Perceptron()
perceptron.fit(X_train, y_train_encoder)

# (100,) mean 1 hidden layer 100 neuron
base_mlpClassifer = MLPClassifier(hidden_layer_sizes=(100,),activation='logistic',solver='sgd')
base_mlpClassifer.fit(X_train, y_train_encoder)

topMLP = MLPClassifier()
gd_parma=[
    { 'activation': ['logistic','tanh','relu','identity'], 'hidden_layer_sizes':[(30,50),(10,10,10)], 'solver':['adam','sgd']}
]
gs = GridSearchCV(topMLP, param_grid=gd_parma)
gs.fit(X_train, y_train_encoder)



# function to write the output to file
def writeTofile(model,mode, confmatrix,class_report):
    file_name ="./drugs-performance.txt" 
    
    with open(file_name, mode) as textfile:  
        textfile.writelines("(a) "+model)  
        textfile.writelines("\n(b) The confusion matrix\n")
        for line in confmatrix:
            textfile.writelines("\t"+ '\t'.join([str(x) for x in line])+"\n")
        textfile.writelines("\n(c) The precision, recal and F1-measure\n")
        textfile.writelines("\n(d) The accuracy, macro-average F1 and Weigh-average F1\n")
        textfile.writelines(class_report)
        textfile.writelines("\n")
    textfile.close()


def extractValueWriteToFile( modelname,mode,y_pred_class,i):
    confmatrix = confusion_matrix(le.transform(y_test), y_pred_class)
    class_report =classification_report(le.transform(y_test), y_pred_class)
    if(i != 0 ):
        class_report_dict =classification_report(le.transform(y_test), y_pred_class,output_dict=True)
        allreport[modelname][acustr].append(round(class_report_dict[acustr],4) )
        allreport[modelname][macrostr].append(round(class_report_dict['macro avg'][f1str],4))
        allreport[modelname][weistr].append(round(class_report_dict['weighted avg'][f1str],4))
        
    # acc_score = accuracy_score(y_test, y_pred_class)
    writeTofile(modelname, mode ,confmatrix,class_report)


for i in range(11):
    y_pred_class= multinomialNB.predict(X_test)
    extractValueWriteToFile("NB",'w',y_pred_class,i)

        # the decision tree base
    y_pred_class= decisionTreeBase.predict(X_test)
    extractValueWriteToFile("Base-DT",'a',y_pred_class,i)

        # the decsion top using grid search
    y_pred_class= gd_dctree.predict(X_test)
    extractValueWriteToFile("Top-DT",'a',y_pred_class,i)

        # the perceptron
    y_pred_class= perceptron.predict(X_test)
    extractValueWriteToFile("PER",'a',y_pred_class,i)

        # the base MLT
    y_pred_class= base_mlpClassifer.predict(X_test)
    extractValueWriteToFile("Base-MLP",'a',y_pred_class,i)

        # the Top MLP using grid search
    y_pred_class= gs.predict(X_test)
    extractValueWriteToFile("Top-MLP",'a',y_pred_class,i)


cv_result ={}
cv_result["NB"]= cross_validate(multinomialNB, X, le.transform(y), cv=10 , scoring=['accuracy','f1_macro','f1_weighted'])
cv_result["Base-DT"] = cross_validate(decisionTreeBase, X, le.transform(y), cv=10 , scoring=['accuracy','f1_macro','f1_weighted'])
cv_result["Top-DT"]= cross_validate(gd_dctree, X, le.transform(y), cv=10 , scoring=['accuracy','f1_macro','f1_weighted'])
cv_result["PER"]= cross_validate(perceptron, X, le.transform(y), cv=10 , scoring=['accuracy','f1_macro','f1_weighted'])
cv_result["Base-MLP"] = cross_validate(base_mlpClassifer, X, le.transform(y), cv=10 , scoring=['accuracy','f1_macro','f1_weighted'])
cv_result["Top-MLP"] = cross_validate(gs, X, le.transform(y), cv=10 , scoring=['accuracy','f1_macro','f1_weighted'])

def extendReport(dict,type,prefix):
    with open("./drugs-performance.txt" , 'a') as tf: 
        tf.writelines("\n\n"+type)
        for key, value in dict.items(): 
            tf.writelines("\n"+key+"\n")
            tf.writelines("\taverage accuracy "+str(round(np.mean(value[prefix+'accuracy']),4))+"\n")
            tf.writelines("\taverage macro-average F1 "+str(round(np.mean(value[prefix+'f1_macro']),4))+"\n")
            tf.writelines("\taverage weighted-average F1 "+str(round(np.mean(value[prefix+'f1_weighted']),4))+"\n")
            tf.writelines("\tstandard deviation accuracy "+str(round(np.std(value[prefix+'accuracy']),4))+"\n")
            tf.writelines("\tstandard deviation macro-average F1 "+str(round(np.std(value[prefix+'f1_macro']),4))+"\n")
            tf.writelines("\tstandard deviation weighted-average F1 "+str(round(np.std(value[prefix+'f1_weighted']),4))+"\n")
        tf.close()

extendReport(allreport,"The same train test set re run 10 time",'')
extendReport(cv_result,"Cross Validation CV 10",'test_')
