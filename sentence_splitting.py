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
  
def read_data(filename):
    d_frame = pd.read_csv(filename, sep=';', encoding='ISO-8859-15')
    x_data= d_frame['tiatxv']

    y_data = d_frame['itclabel']

    data = pd.concat([x_data, y_data], axis=1, ignore_index=True)
    data = data.dropna(axis=0)

    data =  np.matrix(data)
    
    return data
