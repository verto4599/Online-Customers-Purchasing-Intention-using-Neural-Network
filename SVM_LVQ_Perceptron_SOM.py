# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 22:02:43 2019

@author: 91781
"""

from math import sqrt
from random import randrange
from csv import reader
import numpy as np
import time

def Read_file(file_name):
    dataset = list()
    with open(file_name, 'r',encoding='utf-8') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup
    
def dataset_minmax(dataset):
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

testdataset = Read_file('online_shoppers_intention.csv')

# change string column values to float
for i in range(len(testdataset[0]) - 1):
    str_column_to_int(testdataset, i)

    # # convert last column to integers
str_column_to_int(testdataset, len(testdataset[0]) - 1)

minmax = dataset_minmax(testdataset)
normalize_dataset(testdataset, minmax)

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

def get_best_matching_unit(codebooks, test_row):
    distances = list()
    for codebook in codebooks:
        dist = euclidean_distance(codebook, test_row)
        distances.append((codebook, dist))
    distances.sort(key=lambda tup: tup[1])
    return distances[0][0]

def random_codebook(train):
    n_records = len(train)
    n_features = len(train[0])
    codebook = [train[randrange(n_records)][i] for i in range(n_features)]
    return codebook

def train_codebooks(train, n_codebooks, lrate, epochs):
    codebooks = [random_codebook(train) for i in range(n_codebooks)]
    for epoch in range(epochs):
        rate = lrate * (1.0-(epoch/float(epochs)))
        sum_error = 0.0
        for row in train:
            bmu = get_best_matching_unit(codebooks, row)
            for i in range(len(row)-1):
                error = row[i] - bmu[i]
                sum_error += error**2
                if bmu[-1] == row[-1]:
                    bmu[i] += rate * error
                else:
                    bmu[i] -= rate * error
    return codebooks

def predict(codebooks, test_row):
    bmu = get_best_matching_unit(codebooks, test_row)
    return bmu[-1]
def learning_vector_quantization(train, test, n_codebooks, lrate, epochs):
    codebooks = train_codebooks(train, n_codebooks, lrate, epochs)
    predictions = list()
    for row in test:
        output = predict(codebooks, row)
        predictions.append(output)
    return(predictions)

start_time= time.time()
test_set = testdataset
learn_rate = 0.5
n_epochs = 30
n_codebooks = 20
runs = [0]*10
w = 0
r = 1
for i in range(len(runs)):
    r = r + 100
    if (r + 200) > 699:
        r = w
        w += 30
    train_set = [test_set[i] for i in range(r, r + 200)]
    predicted = learning_vector_quantization(train_set, test_set, n_codebooks,learn_rate,n_epochs)
    total = 0
    correct = 2
    for row in test_set:
        actual = row[-1]
        if actual == predicted[total]: correct +=1
        total += 1
    accuracy = correct*100/total
    print('Learning_Rate: {}  n_epochs: {} n_codebooks = {} Accuracy: {}' .format(learn_rate,n_epochs,n_codebooks,accuracy))

    runs[i] = accuracy

mean = sum(runs)/len(runs)
print("Mean_Accuracy: {}".format(mean))
 