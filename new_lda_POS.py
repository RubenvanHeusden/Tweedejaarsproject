from __future__ import division
from nltk.tokenize import RegexpTokenizer
from nltk.tag import PerceptronTagger
from nltk.corpus import alpino as alp
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
from gensim import corpora, models
import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
import numpy as np
import pyLDAvis.gensim
import json
import warnings
from collections import Counter
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity


import pandas as pd



tokenizer = RegexpTokenizer(r'\w+')

# create Dutch POStagger

training_corpus = list(alp.tagged_sents())
tagger = PerceptronTagger(load=True)
tagger.train(training_corpus)


# create Dutch stop words list
en_stop = get_stop_words('nl')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()



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
print "data read"

# list for tokenized documents in loop
texts = []

# loop through document list to:
# 1) clean and tokenize document string
# 2) remove stop words from tokens
# 3) stem tokens
for i in doc_set:
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    tagged = tagger.tag(tokens)

    important_tokens = []
    wordfilter = [u'noun', u'verb', u'adj']

    for word in tagged:
        if word[1] in wordfilter:
            important_tokens.append(word[0])

#    stopped_tokens = [i for i in tokens if not i in en_stop]
#    stemmed_tokens = [p_stemmer.stem(i) for i in important_tokens]
    texts.append(important_tokens)


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

print "pre processing completed"
# generate LDA model

aLdaModel = LdaModel(corpus=corpus, id2word=dictionary, iterations=50, num_topics=20)
cm = CoherenceModel(model=aLdaModel, texts=truncated_texts, dictionary=dictionary, coherence='c_v')

print cm.get_coherence()


print(aLdaModel.print_topics(num_topics=20, num_words=10))
