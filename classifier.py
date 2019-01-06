import math
import random
import numpy as np
from hw3_utils import abstract_classifier
from hw3_utils import abstract_classifier_factory

class knn_factory(abstract_classifier_factory):
    def __init__(self,k):
        self.k = k

    def train(self, data, labels):
        print("train called")
        classifier = knn_classifier(data, labels, self.k)
        return classifier


class knn_classifier(abstract_classifier):
    def __init__(self, train_data, train_labels, k=1):
        self.data = train_data
        self.labels = train_labels
        self.num_samples = len(train_labels)
        self.k = k
        self.calls = 0

    def classify(self, features):
        print("classify called ", self.calls)
        self.calls += 1
        # pass on all data and generate dist for all.
        distances = []
        for sample in self.data:
            dist = euclidean_distance(features, sample)
            distances.append(dist)

        # select k nearest point
        idx = np.argpartition(distances, self.k)
        nearest_k_idx = idx[:self.k]

        # get list of labels of nearest points.
        k_labels = [self.labels[i] for i in nearest_k_idx]

        # return the majority prediction.
        pred_true = k_labels.count(True)
        pred_false = k_labels.count(False)

        if pred_true >= pred_false:
            return True
        elif pred_true < pred_false:
            return False


def euclidean_distance(feature_list1, feature_list2):
    sum = 0
    for i in range(len(feature_list1)):
        x = feature_list1[i]
        y = feature_list2[i]
        sum += (x-y)**2
    #dist = math.sqrt(sum)
    dist = sum
    return dist


def evaluate(classifier_factory, k):
    # k - number of folds.

    # TODO: get k-folded dataset.
    # TODO: run k-cross validation.
    # TODO: 1 set is chosen as test.
    # TODO: all others are chosen as train.
    # TODO: repeat until all k fold been as "test set".

    global_acc = 0
    global_err = 0
    for iter_k in range(k):
        #print("test fold is - ",iter_k)
        test_data, test_labels = load_k_fold_data(iter_k)

        train_data = []
        train_labels = []
        # Get all train data and labels
        for fold_idx in range(k):
            if fold_idx == iter_k:
                continue # dont train on test set.
            data_temp, labels_temp = load_k_fold_data(fold_idx)
            train_data += data_temp
            train_labels += labels_temp
            #print("added to train fold - ",fold_idx)

        classifier = classifier_factory.train(train_data, train_labels)

        classifications = []
        for sample in test_data:
            classifications.append(classifier.classify(sample))

        local_err = 0
        local_acc = 0
        N = len(classifications)
        for i in range(N):
            if classifications[i] == test_labels[i]:
                local_acc += 1
            else:
                local_err += 1
        global_err += local_err/N
        global_acc += local_acc/N

    AvgErr = global_err/k
    AvgAcc = global_acc/k

    return AvgErr, AvgAcc


def load_k_fold_data(idx):
    train_i = []
    labels_i = []
    # loading an ecg_fold_<idx>.data
    filename = "ecg_fold_" + str(idx) + ".data"
    file = open(filename, "r")

    local_features = []
    while True:
        line = file.readline()
        if line == '':
            # EOF
            #print("EOF")
            break
        elif line == 'True\n':
            #print("adding True label")
            labels_i.append(True)
            train_i.append(np.array(local_features))
            local_features = []
        elif line == 'False\n':
            #print("adding False label")
            labels_i.append(False)
            train_i.append(np.array(local_features))
            local_features = []
        else:
            #print("Adding feature")
            local_features.append(float(line))
    return train_i, labels_i

