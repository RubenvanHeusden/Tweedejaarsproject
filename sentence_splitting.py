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
import re


# Reading in the data and dropping the entries in which the feedback was left
# blank.
  
def read_data_filtered(filename):
    d_frame = pd.read_csv(filename, sep=';', encoding='ISO-8859-15')
    x_data= d_frame['tiatxv']
    y_data = d_frame['itclabel']
    data = pd.concat([x_data, y_data], axis=1, ignore_index=True)
    data = data.dropna(axis=0)
    
    new = []
    for i in range(0,data.shape[0]) : 
        row = data.iloc[i]
        line = row[0].lower()
        label = row[1]
        splitted = re.split(r'[\n.;,\?\!]',line)
        for item in splitted : 
            item = re.sub(r'[^\x00-\x7F]+','', item)
            item = re.sub(ur"[-/]", " ",item)
            new.append([item,label])
        
    data = np.matrix(new)
    print data.shape
    return data
    
    


data = read_data_filtered('small_sample_data.csv')

### STEP 1 : SPLITTING THE DATA AT NEWLINES AND OTHER CHARACTERS ###

# Convert everything into lowercase 
lines = [cell.item(0).lower() for cell in data[:,0]]

tokenized = []

for line in lines : 
    tokens = re.split(r'[\n.;,\?\!]',line)
    tokenized+=tokens
    
    
    
    # Daan hier moeten we nog aan gaan werken
    # kan wss ook wel allemaal in een for loop 
    
    
### STEP 2 : REMOVING ANY STRANGE CHARACTERS IN STRINGS : ###
    # a preliminary test is made allowing only ascii upper and lowercase letters    
    
    
for x in range (0,len(tokenized)) : 
    # filtered unicdoe seperately , strange stuff was happening
    tokenized[x] = re.sub(r'[^\x00-\x7F]+','', tokenized[x])
    tokenized[x] = re.sub(ur"[-/]", " ",tokenized[x])
    # See if you can think of more characters that should be filtered out 

    
