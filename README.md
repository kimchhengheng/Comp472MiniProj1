# Comp472MiniProj1
https://github.com/kimchhengheng/Comp472MiniProj1

The code for this mini project are seperate into two file
1. Task1.py
* Using the dataset load_files to load the file
* Split the data to train set and test set
* Using CountVectorizer to extract the feature and frequency of it
* Create the model
* Use model to predict the data and generate the confusing matrix and classification report
* Write to text file
2. Task2.py
* Using Panda to reach data from CSV
* Convert the ordinal and nominal feature into numberical value by using the panda.get_dummies and panda.Categorical
* Split the data to train set and test set
* Encode the target value by using LabelEncoder
* Create the Model, some models used GridSearchCV
* Use model to predict the data and generate the confusing matrix and classification report
* Write to text file
* Compute the average to running the program 10 time and append to text file

To run these python code, 
* Make sure you have the dataset in the same directory, or update the path to the dataset in the .py file
* Make sure you python interpreter are properly set
* Make sure the necessary library is install such as sklearn, panda, numpy
* For Task1.py, it could be run by terminal or click run button on the preffered IDE. However, running all the code in one time could lead the same result since the number of feature is the same. Therefore, you can comment out some part does not need and run it individuallly.
* For Task2.py, it could be run by terminal or click run button on the preferred IDE
For task2.py, in order to run the cross validation for part 8 of task 2 , it would take more than 7 minute to finish the program

Student name: kimchheng heng, Id 26809413
