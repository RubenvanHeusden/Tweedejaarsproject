from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import csv
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


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
    
    def predict(self,data_matrix,test_set_size) : 
        # Retrieving the last 3 colums from the data matrix 
        x_data = np.concatenate((data_matrix[:,3],data_matrix[:,4],
        data_matrix[:,5]),axis=0)
        
        # Constructing the y data by labeling the consecutive classes
        y_data =  np.repeat([0,1,2],data_matrix.shape[0])
        y_data =  np.reshape(y_data,(x_data.shape[0],1))
        
        # Merge the x and y data into 1 matrix
        complete_data = np.concatenate((x_data,y_data),axis=1)
        
        # Shuffling the data
        shuffled_data = train_test_split(complete_data,test_size=test_set_size)
        
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
        text_clf = Pipeline([('vect', CountVectorizer(input = 'content')),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf',MultinomialNB() ),])

        text_clf = text_clf.fit(x_train_data, np.ravel(y_train_data))
        predicted = text_clf.predict(x_test_data)
        
        return accuracy_score(y_test_data,predicted,normalize = True)
