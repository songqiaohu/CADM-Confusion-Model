import matplotlib.pyplot as plt
import skmultiflow as sk
import numpy as np
import time
import random
import warnings
from skmultiflow.data import SEAGenerator, WaveformGenerator
from skmultiflow.data import DataStream
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.meta import DynamicWeightedMajorityClassifier
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.bayes import NaiveBayes
from skmultiflow.drift_detection.adwin import ADWIN
from Def_OSELM import OSELMClassifier
from Def_BLSWoodbury import BLS_Woodbury
import pandas as pd

#########Parameter setting###########
train_size = 200
chunk_size = 200
label_ratio = 0.2
classifier_selection = 'DWM'  #classifier
############model selection ############
def classifier_select(string = 'NB'):
    if string == 'NB':
        return NaiveBayes()
    if string == 'HT':
        return HoeffdingTreeClassifier()
    if string == 'OSELM':
        return OSELMClassifier()
    if string == 'BLS':
        return     BLS_Woodbury(Nf=10,
                            Ne=10,
                            N1=10,
                            N2=10,
                            map_function='sigmoid',
                            enhence_function='sigmoid',
                            reg=0.001)     #BLS
    if string == 'ARF':
        return AdaptiveRandomForestClassifier()
    if string == 'DWM':
        return DynamicWeightedMajorityClassifier()

accuracy_ten = []
for i in range(10):
    print(i)
    classifier = classifier_select(classifier_selection)
    dataset = 'doubleline'
    filename = './data/' + dataset + '_100000.csv'
    dataframe = pd.read_csv(filename, header = None)
    dim = dataframe.shape
    array = dataframe.values
    cosine_all = []
    right = 0
    prediction = 0
    Y = array[0: 100000, dim[1] - 1]
    X = array[0: 100000, 0:dim[1] - 1]
    stream = DataStream(X, Y)


    # Setup Classifier


    x, y = stream.next_sample(train_size)
    index = np.random.permutation(np.arange(train_size))
    x = x[index]
    y = y[index]
    X_train = x[0 : round(train_size * label_ratio)]
    Y_train = y[0 : round(train_size * label_ratio)]

    classifier.fit(X_train, Y_train)

    # Setup variables to control loop and track performance
    count = 0
    n_samples = chunk_size
    correct_cnt_classifier = 0
    max_samples = 100000

    result_classifier = []
    result = []
    accuracy_classifier = []

    while n_samples < max_samples and stream.has_more_samples():

        x, y = stream.next_sample(chunk_size)


        y_pred_classifier = classifier.predict(x)

        accuracy = 0

        for i in range(len(y)):
            if y[i] == y_pred_classifier[i]:
                correct_cnt_classifier += 1



        index = np.random.permutation(np.arange(chunk_size))
        x = x[index]
        y = y[index]
        X_update = x[0 : round(label_ratio * chunk_size)]
        Y_update = y[0 : round(label_ratio * chunk_size)]
        classifier.partial_fit(X_update, Y_update)


        n_samples += 200
        count += 200

        if count % (max_samples * 0.10) == 0:
            print('Have processed {:.2f}%'.format(count/max_samples), 'samples')


    print('{} accuracy: {}'.format(classifier_selection, correct_cnt_classifier / (n_samples - 200)), correct_cnt_classifier)
    accuracy_ten.append(correct_cnt_classifier / n_samples)
print(accuracy_ten)
print(np.mean(accuracy_ten), '  ', np.std(accuracy_ten))
