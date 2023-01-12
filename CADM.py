from skmultiflow.bayes import NaiveBayes
from skmultiflow.data import DataStream
import pandas as pd
import numpy as np
import copy
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import csv
from Def_OSELM import OSELMClassifier
from Def_BLSWoodbury import BLS_Woodbury
from skmultiflow.trees import HoeffdingTreeClassifier
import random
###### experiment parameter set ##########
train_size = 200
chunk_size = 200
label_ratio = 0.2
class_count = 2
max_samples = 100000
classifier = 'BLS'  # HT, NB, OSELM, BLS
######## function definition ###############
######## stream generation #########################
def data_stream(X, Y):
    stream = DataStream(X, Y)
    return stream

######## labeled, unlabeled###############
def data_dividing(X, Y, method = 0):
    size = X.shape[0]
    label_size = round(size * label_ratio)
    if method == 0:

        X_labeled = X[0:label_size]
        Y_labeled = Y[0:label_size]
        X_unlabeled = X[label_size::]
        Y_unlabeled = Y[label_size::]
    return X_labeled, Y_labeled, X_unlabeled, Y_unlabeled

####### pseudo label #####################
def pseudo_label(X, classifier, size):  #get hard pseudo labels
    Y_pseudo = np.zeros(size)
    for i in range(size):
        Y_pseudo[i]=classifier.predict(np.array([X[i]]))
    return X[0:size,:], Y_pseudo


########cosine similarity ###############
def cosine_similarity(classifier1, classifier2, X_chunk):
    h1 = []
    h2 = []
    for i in range(X_chunk.shape[0]):
        h1.append(classifier1.predict_proba([X_chunk[i, :]])[0])
        h2.append(classifier2.predict_proba([X_chunk[i, :]])[0])
    sum_cos = 0
    for j in range(class_count):
        h11 = np.array([i[j] for i in h1])
        h22 = np.array([i[j] for i in h2])
        sum_cos += h11.dot(h22) / (np.linalg.norm(h11) * np.linalg.norm(h22))
        if classifier == 'OSELM':
            sum_cos *= 2
            break
    return sum_cos / class_count, h11 ,h22

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



def main():

    ########  data stream  ##################
    dataset = 'doubleline'
    filename = './data/' + dataset + '_100000.csv'
    dataframe = pd.read_csv(filename, header = None)
    dim = dataframe.shape
    array = dataframe.values
    cosine_all = []
    right = 0
    prediction = 0
    Y = array[0 : max_samples, dim[1] - 1]
    X = array[0 : max_samples, 0 : dim[1] - 1]
    stream = data_stream(X, Y)

########  base classifier generation ###############


    classifier_1 = classifier_select(classifier)


    X_train, Y_train = stream.next_sample(train_size)
    index = np.random.permutation(np.arange(train_size))
    X_train = X_train[index]
    Y_train = Y_train[index]
    X_labeled, Y_labeled, X_unlabeled, Y_unlabeled = data_dividing(X_train, Y_train)
    classifier_1.fit(X_labeled, Y_labeled)

    X_update, Y_update = stream.next_sample(chunk_size)
    index = np.random.permutation(np.arange(chunk_size))
    X_update = X_update[index]
    Y_update = Y_update[index]
    X_labeled, Y_labeled, X_unlabeled, Y_unlabeled = data_dividing(X_update, Y_update)
    X_pseudo, Y_pseudo = pseudo_label(X_unlabeled, classifier_1, X_labeled.shape[0])
    classifier_2 = copy.deepcopy(classifier_1)
    classifier_2.partial_fit(np.row_stack((X_pseudo, X_labeled)), np.append(Y_pseudo, Y_labeled))


    #######  chunk prediction ###############
    n_samples = 2 * chunk_size   # 2 * chunk_size samples passed
    Y_prediction_all = []
    Y_real_all = []
    accuracy_track = []
    window = []
    threshold = []  #to store all thresholds
    drift = []
    while n_samples <= max_samples and stream.has_more_samples():
        n_samples += chunk_size
        X_chunk, Y_chunk = stream.next_sample(chunk_size)

        ################ drift detection ################################
        cosine, _, _ = cosine_similarity(classifier_1, classifier_2, X_chunk)
        print('chunk: ', n_samples / chunk_size, 'n_samples: ', n_samples, 'cosine: ', cosine)
        cosine_all.append(cosine)

        ##### update window ############
        window.append(cosine)
        if len(window) > 10 :
            window.pop(0)

        threshold.append(np.mean(window) - 2 * np.sqrt(np.var(window)))

        if cosine < threshold[len(threshold) - 1] :
            drift.append(int(n_samples / 200))
            print('drift occurs!!')
            index = np.random.permutation(np.arange(X_chunk.shape[0]))
            X_chunk = X_chunk[index]
            Y_chunk = Y_chunk[index]
            ########## select some samples to annotate in new chunk ########################
            X_labeled, Y_labeled, _, _ = data_dividing(X_chunk, Y_chunk)

            ##########    retrain   #########################
            classifier_1 = classifier_select(classifier)

            classifier_1.fit(X_labeled, Y_labeled)
            classifier_2 = copy.deepcopy(classifier_1)
            window = []

            ############ predict ############################################
            Y_prediction = classifier_2.predict(np.array(X_chunk))

            for i in Y_prediction:
                Y_prediction_all.append(i)
            for j in Y_chunk:
                Y_real_all.append(j)
            accuracy_count = 0

            for i in range(len(Y_prediction)):
                prediction += 1
                if Y_prediction[i] == Y_chunk[i]:
                    right += 1
                    accuracy_count += 1

            accuracy_track.append(accuracy_count / len(Y_prediction))


        else:
            ############ predict #############################
            Y_prediction = classifier_2.predict(X_chunk)
            for i in Y_prediction:
                Y_prediction_all.append(i)
            for j in Y_chunk:
                Y_real_all.append(j)
            accuracy_count = 0
            for i in range(len(Y_prediction)):
                prediction += 1
                if Y_prediction[i] == Y_chunk[i]:
                    right += 1
                    accuracy_count += 1
            accuracy_track.append(accuracy_count / len(Y_prediction))
            index = np.random.permutation(np.arange(X_chunk.shape[0]))
            X_chunk = X_chunk[index]
            Y_chunk = Y_chunk[index]
            ########## select samples to annotate in new chunk #####################
            X_labeled, Y_labeled, X_unlabeled, Y_unlabeled = data_dividing(X_chunk, Y_chunk)

            ################# get pseudo labels ###################################
            X_pseudo, Y_pseudo = pseudo_label(X_unlabeled, classifier_2, X_labeled.shape[0])#the sizes are equal

            ############ update  ####################################
            classifier_1 = copy.deepcopy(classifier_2)
            classifier_2.partial_fit(np.row_stack((X_labeled, X_pseudo)), np.append(Y_labeled, Y_pseudo))

    print('cosine similarity: ', cosine_all)
    print('drifts occur at: ', drift)

    plt.figure(1)
    p1,  = plt.plot(range(3, len(cosine_all) + 3), cosine_all, 'b', label = 'cosine similarity')

    ####### where drifts occur ###############################
    y_drift = np.linspace(min(cosine_all), max(cosine_all), 100)
    x_drift = [[25 * j for i in y_drift] for j in range(1, 20)]
    p2 = _
    for i in x_drift:
        p2,  = plt.plot(i, y_drift, 'y--', label = 'concept drift')

    p3,  = plt.plot(range(2, len(cosine_all) + 2), threshold, 'r--', label = 'threshold')
    plt.legend(handles = [p1, p2, p3], labels = ['cosine similarity', 'concept drift', 'threshold'])
    plt.xlabel('chunk')
    plt.ylabel('cosine similarity')
    plt.title('In ${}$, drift occurs every 25 chunks'.format(dataset))
    plt.gcf().subplots_adjust(left=None, top=None, bottom=None, right=None)

    plt.figure(2)
    plt.plot(range(1, len(accuracy_track) + 1), accuracy_track, color = 'forestgreen')
    plt.xlabel('$chunk$')
    plt.ylabel('$accuracy\ of\ each\ chunk$')
    plt.title(
        '$CADM-{},\ overall accuracy\ =\ {}\%$'.format(classifier, round(right / prediction * 100, 3)))

    print('overall accuracy = {}%'.format(right / prediction * 100))
    plt.show()



if __name__ == "__main__" :
    main()