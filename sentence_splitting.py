from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import csv
from nltk.corpus import stopwords
from nltk.stem.snowball import DutchStemmer


# Reading in the data and dropping the entries in which the feedback was left
# blank.
  
def read_data(filename):
    d_frame = pd.read_csv(filename, sep=';', encoding='ISO-8859-15')
    x_data= d_frame['tiatxv']

    y_data = d_frame['itclabel']

    data = pd.concat([x_data, y_data], axis=1, ignore_index=True)
    data = data.dropna(axis=0)

    data =  np.matrix(data)
    
    return data
    
    


data = read_data('small_sample_data.csv')
# Convert everything into lowercase 
lines = [cell.item(0).lower() for cell in data[:,0]]

