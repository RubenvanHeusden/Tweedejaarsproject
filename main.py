import csv
import numpy as np


data = []
with open('sample_data.csv', 'rU') as csvfile:
    
    spamreader = csv.reader(csvfile, delimiter=';')
    spamreader.next()
    for row in spamreader:         
        data.append(row)


data_matrix = np.matrix(data)
print data_matrix
