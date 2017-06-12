import csv

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



data = read_data(DATA_CSV)
people = get_people(data)
feedback_dict = group_feedback(data, people)



print '---reading in and formatting data: complete---'

print feedback_dict


###############################################################################
