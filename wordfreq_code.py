from __future__ import division
import csv
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np

###############################################################################
                    # GLOBAL VARIABLES

DATA_CSV = 'language_filtered_data.csv'

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
	if data[i][3] == 'KEEP':
            keep_list.append(data[i][4])
        elif data[i][3] == 'START':
            start_list.append(data[i][4])
        else:
            stop_list.append(data[i][4])

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
        words.extend(sentence.split())

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

# Counter for different sets.
keep_counter = Counter(keep_training_words)
start_counter = Counter(start_training_words)
stop_counter = Counter(stop_training_words)
all_counter = Counter(all_words)

print all_counter
