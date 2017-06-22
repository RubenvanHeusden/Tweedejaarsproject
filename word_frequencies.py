import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import math


data_file = 'sample_data.csv'
csv_output_file = 'frequencies_with_words.csv'
output_to_csv = False
plot_FoF = True

# Sentence preprocessing
to_ascii = True # Required for CSV output
lower = True
punctuation = True #To be added


# Read the data from the file
def read_data(filename):
    d_frame = pd.read_csv(filename, sep=';', encoding='ISO-8859-15')
    x_data= d_frame['tiatxv']
    y_data = d_frame['itclabel']
    data = pd.concat([x_data, y_data], axis=1, ignore_index=True)
    data = data.dropna(axis=0)
    data =  np.matrix(data)
    return data

def reverse_list(list_input):
	length = len(list_input)
	result = [0 for i in range(length)]
	for i in range(length):
		result[i] = list_input[length-1-i]
	
	return result

data = read_data(data_file)


# Calculate the frequency of each word
word_frequencies = {}
for elem in data[:,0]:
	sentence = elem.item()
	if to_ascii:
		sentence = sentence.encode('ascii', 'ignore')
	if lower:
		sentence = sentence.lower()
	
	sentence = sentence.split()
	for word in sentence:
		if word_frequencies.has_key(word):
			word_frequencies[word] += 1
		else:
			word_frequencies[word] = 1


# Find the words belonging to each frequency
word_frequencies_inv = {}
for word in word_frequencies.keys():
	freq = word_frequencies[word]
	if word_frequencies_inv.has_key(freq):
		word_frequencies_inv[freq].append(word)
	else:
		word_frequencies_inv[freq] = [word]


# Build a sorted list of the frequencies in decreasing order
frequencies = word_frequencies_inv.keys()
frequencies.sort()
frequencies.reverse()


# Print the top n frequencies and the words that have these frequencies
#n_printed = 100
#for frequency in frequencies[:n_printed]:
#	print('{} : {}'.format(frequency, word_frequencies_inv[frequency]))



# Output the FoF in a plot
if plot_FoF:
	freq_reverse = reverse_list(frequencies)
	plt.plot([math.log(item) for item in freq_reverse], [math.log(len(word_frequencies_inv[freq])) for freq in freq_reverse])
	plt.show()



# Print the data to a csv file
if output_to_csv:
	with open(csv_output_file, 'wb') as f:
		w = csv.writer(f)
		w.writerow(['Frequency', 'Words'])
		for frequency in frequencies:
			w.writerow([frequency] + word_frequencies_inv[frequency])

