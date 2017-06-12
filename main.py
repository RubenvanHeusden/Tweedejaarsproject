from __future__ import division
import csv
from nltk.tokenize import word_tokenize
from collections import Counter

###############################################################################
                    # GLOBAL VARIABLES

DATA_CSV = 'vbdata.csv'

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



# get_people extracts the distinctive person ID's from the data
def get_people(data):
    people = []

    for i in range(0, len(data)):
        people.append(data[i][0])

    people = list(set(people))

    return people



# group_feedback groups the feedback per person and stores this in a dictionary.
# Key: the person ID number, Data: the corresponding feedback entries given in
# a list format. (Person ID: [Feedback_1, Feedback_2, Feedback_3])
def group_feedback(data, people):
    feedback_dict = {}
    x = 0

    for i in range(0, len(people)):
        person = people[i]
        feedback_dict[person] = []

        for j in range(x, len(data)):
            if data[j][0] == person:
                feedback_dict[person].append(data[j])
            else:
                x = j
                break

    return feedback_dict



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


length = len(feedback_list[0])

keep_sets = split_training_data(feedback_list[0], length)
start_sets = split_training_data(feedback_list[1], length)
stop_sets = split_training_data(feedback_list[2], length)


def tokenize_set(dataset):
    words = []
    for sentence in dataset:
        words.extend(word_tokenize(sentence))

    return words

keep_training_words = tokenize_set(keep_sets[0])
start_training_words = tokenize_set(start_sets[0])
stop_training_words = tokenize_set(stop_sets[0])


def get_word_probabilities(words):
    word_probabilities = {}
    word_freq = Counter(words)
    amount_of_words = len(words)
    distinctive_words = list(set(words))
    for word in distinctive_words:
        word_probabilities[word] = word_freq[word] / amount_of_words

    return word_probabilities


wp_keep_training = get_word_probabilities(keep_training_words)
wp_start_training = get_word_probabilities(start_training_words)
wp_stop_training = get_word_probabilities(stop_training_words)





print '---sets created---'
