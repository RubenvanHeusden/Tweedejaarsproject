from __future__ import division
import csv
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np

###############################################################################
                    # GLOBAL VARIABLES

DATA_CSV = 'testdata.csv'

def settings():
    print 'Settings:\n'
    print 'Data file:   ' + DATA_CSV

settings()


###############################################################################

                    # DATA READING AND FORMATTING



# read_data reads in the free input feedback data from the csv file.
def read_data(data_csv):
    data = []

    with open(data_csv, 'rU') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        spamreader.next()

        for row in spamreader:
            data.append(row)

    return data



def feedback_types(data):
    keep_list = []
    start_list = []
    stop_list = []

    for i in range(0, len(data)):
        keep_list.append(data[i][3])
        start_list.append(data[i][4])
        stop_list.append(data[i][5])

    return (keep_list, start_list, stop_list)



def split_training_data(data_list, length):
    training_length = round( length * 0.8)
    training_length = int(training_length)
    return (data_list[0:training_length], data_list[training_length:length])



data = read_data(DATA_CSV)
feedback_list = feedback_types(data)



print '---reading in and formatting data: complete---'



###############################################################################

def tokenize_set(dataset):
    words = []
    for sentence in dataset:
        words.extend(word_tokenize(sentence))

    return words



length = len(feedback_list[0])

keep_sets = split_training_data(feedback_list[0], length)
start_sets = split_training_data(feedback_list[1], length)
stop_sets = split_training_data(feedback_list[2], length)

keep_training_words = tokenize_set(keep_sets[0])
start_training_words = tokenize_set(start_sets[0])
stop_training_words = tokenize_set(stop_sets[0])
all_words = keep_training_words + start_training_words + stop_training_words



print '---sets created---'



###############################################################################



# add_one_estimate uses naive bayes estimation with add one smoothing.
# Because a priori the classes are all equally likely, the prior chance
# is neglected.
def add_one_estimate(feedback, ngram_count, class_ngram_count, class_size):
    feedback_tokenized = word_tokenize(feedback)
    sentence_probability = 1

    for ngram in feedback_tokenized:
        if ngram in ngram_count:
            if ngram in class_ngram_count:
                sentence_probability *= ((class_ngram_count[ngram] + 1)
                                         / (class_size + 3))
            else:
                sentence_probability *= 1 / (class_size + 3)
        else:
            sentence_probability *= 1 / 3

    return sentence_probability



def get_class(feedback,
            wc_all_words,
            wc_keep_training,
            wc_start_training,
            wc_stop_training,
            keep_size,
            start_size,
            stop_size):
    p_keep = add_one_estimate(feedback, wc_all_words, wc_keep_training, keep_size)
    p_start = add_one_estimate(feedback, wc_all_words, wc_start_training, start_size)
    p_stop = add_one_estimate(feedback, wc_all_words, wc_stop_training, stop_size)

    return np.argmax([p_keep, p_start, p_stop])



wc_keep_training = Counter(keep_training_words)
wc_start_training = Counter(start_training_words)
wc_stop_training = Counter(stop_training_words)
wc_all_words = Counter(all_words)



print '---classifying---'

###############################################################################

def classify_feedback(feedback_list, classnr):
    accurate = 0
    keep_size = len(keep_training_words)
    start_size = len(start_training_words)
    stop_size = len (stop_training_words)
    for feedback in feedback_list:
        prediction = get_class(feedback,
                            wc_all_words,
                            wc_keep_training,
                            wc_start_training,
                            wc_stop_training,
                            keep_size,
                            start_size,
                            stop_size)
        if prediction == classnr:
            accurate += 1

    return accurate / len(feedback_list)

keep_accuracy = classify_feedback(keep_sets[1], 0)
start_accuracy = classify_feedback(start_sets[1], 1)
stop_accuracy = classify_feedback(stop_sets[1], 2)

print 'keep accuracy =      ' + str(keep_accuracy)
print 'start accuracy =     ' + str(start_accuracy)
print 'stop accuracy =      ' + str(stop_accuracy)
print 'average accuracy =   ' + str((keep_accuracy + start_accuracy + stop_accuracy)/3)
