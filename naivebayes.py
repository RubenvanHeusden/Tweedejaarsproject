from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import csv

#############
#   USAGE   #
#############
# To run this class , create an instance of the Naive Bayes class. 
# The predict function return the accuracy of the Naive Bayes classifier on 
# a test set that is constructed by the class itself.



#############
#   INPUTS  #
#############

# The predict funtion takes as argument a data matrix with the START STOP 
# and KEEP columns, the test_set_size is to be a float indicating the percentage
# of the data that is to be used as a test set.

class NaiveBayes() : 

    def preprocess(self,data_matrix) : 
        dels =  []
        for x in range(0,data_matrix.shape[0]):
            if isinstance(data_matrix[x].item(0),float) :
                dels.append(x)
            elif len(data_matrix[x].item(0).split()) <= 1 : 
                dels.append(x)
                
        data_matrix = np.delete(data_matrix,dels,axis=0)               
        return data_matrix


    def __init__(self,preprocessing = True) : 
        self.preprocessing = preprocessing
        
    
    def predict(self,data_matrix,test_set_size) : 
        
        # Proprocessing the data
        if self.preprocessing : 
            data_matrix = self.preprocess(data_matrix)
            
        # Shuffling the data
        shuffled_data = train_test_split(data_matrix,test_size=test_set_size)
  
        # Extracting the featurs after shuffles for training         
        train_data = np.matrix(shuffled_data[0])
        x_train_data = train_data[:,0]
        x_train_data = [cell.item(0) for cell in x_train_data]
        y_train_data = train_data[:,1]
    
        # Extracting the featurs after shuffles for test
        test_data = np.matrix(shuffled_data[1])
        x_test_data = test_data[:,0]
        x_test_data = [cell.item(0) for cell in x_test_data]
        y_test_data = test_data[:,1]


        # Training the classifier
        
        text_clf = Pipeline([('vect', CountVectorizer(encoding='ISO-8859-1')),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf',MultinomialNB() ),])

        text_clf = text_clf.fit(x_train_data, np.ravel(y_train_data))
        predicted = text_clf.predict(x_test_data)
        
        return accuracy_score(y_test_data,predicted,normalize = True)
        
        
        
        
        
        
