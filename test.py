import pandas as pd
import numpy as np
import csv
from naivebayes import NaiveBayes
#from lda import LDA

def read_data(filename) : 

    d_frame = pd.read_csv(filename,sep=';',encoding='ISO-8859-15')

    x_data= d_frame['tiatxv'].values 
    y_data = d_frame["itclabel"].values

    x_data = np.reshape(x_data,(x_data.shape[0],1))
    y_data = np.reshape(y_data,(y_data.shape[0],1))
    data_matrix =  np.matrix(np.concatenate((x_data,y_data),axis=1))

    return data_matrix



data_matrix = read_data('sample_data.csv')
clf = NaiveBayes(preprocessing = True)
print clf.predict(data_matrix,0.2)








