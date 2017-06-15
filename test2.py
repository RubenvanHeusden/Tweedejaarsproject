from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pandas as pd
import numpy as np
from time import time
from nltk.corpus import stopwords

words = stopwords.words('dutch')+stopwords.words('english')

n_samples = 2000
n_topics = 4
n_top_words = 20


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()



def preprocess(data_matrix) : 
    dels =  []
    for x in range(0,data_matrix.shape[0]):
        if isinstance(data_matrix[x].item(0),float) :
            dels.append(x)
        elif len(data_matrix[x].item(0).split()) <= 1 : 
            dels.append(x)
            
    data_matrix = np.delete(data_matrix,dels,axis=0)               
    return data_matrix
    
    
def read_data(filename) : 

    d_frame = pd.read_csv(filename,sep=';',encoding='ISO-8859-15')

    x_data= d_frame['tiatxv'].values 
    y_data = d_frame["itclabel"].values

    x_data = np.reshape(x_data,(x_data.shape[0],1))
    y_data = np.reshape(y_data,(y_data.shape[0],1))
    data_matrix =  np.matrix(np.concatenate((x_data,y_data),axis=1))

    return data_matrix


## old parameters doc_topic_prior = 0.7 topic_word_prior = 0.2 


data_matrix = read_data('sample_data.csv')
data_matrix = preprocess(data_matrix)

# Extracting the featurs after shuffles for training         
train_data = data_matrix[:,0]
train_data = [cell.item(0) for cell in train_data]

tf_vectorizer = CountVectorizer(stop_words=words)

model = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                doc_topic_prior=0.7,
                                topic_word_prior=0.2,
                                random_state=0)
  
tf = tf_vectorizer.fit_transform(train_data)

model.fit(tf)

n_top_words = 20

tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(model, tf_feature_names, n_top_words)













