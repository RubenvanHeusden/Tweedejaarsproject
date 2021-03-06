from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import  matplotlib.pyplot as plt
import numpy as np
import pyLDAvis.gensim
import json
import warnings
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity

from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
import numpy as np
import pandas as pd
from collections import Counter

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('nl')

# Create p_stemmer of class PorterStemmer
#p_stemmer = PorterStemmer()

def read_data(filename):
    d_frame = pd.read_csv(filename, sep=';', encoding='ISO-8859-15')
    x_data= d_frame['tiatxv']

    y_data = d_frame['itclabel']
    data = pd.concat([x_data, y_data], axis=1, ignore_index=True)
    data = data.dropna(axis=0)
    data =  np.matrix(data)
    dels = [x for x in range(0,data.shape[0]) if len(data[x].item(0).strip()) <= 1]

    data = np.delete(data, dels, axis=0)
    return data





data_matrix = read_data('language_filtered_data.csv')
train_data = data_matrix[:,0]
doc_set = [cell.item(0) for cell in train_data]


# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:

    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # stem tokens
    #stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    # add tokens to list
    texts.append(stopped_tokens)


# Truncate very high frequency and low frequency words
feedback_words = [val for sublist in texts for val in sublist]
word_freq = Counter(feedback_words)

truncated_texts = []
for text in texts:
    new_text = []
    for word in text:
        if 500 > word_freq[word] > 4:
            new_text.append(word)
    truncated_texts.append(new_text)



# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(truncated_texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in truncated_texts]

# generate LDA model
#ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes=20)

aLdaModel = LdaModel(corpus=corpus, id2word=dictionary, iterations=50, num_topics=10)

cm = CoherenceModel(model=aLdaModel, texts=texts, dictionary=dictionary, coherence='c_v')



print(aLdaModel.print_topics(num_topics=10, num_words=10))
