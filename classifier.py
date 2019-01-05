import math
import random
import numpy as np
from hw3_utils import abstract_classifier
from hw3_utils import abstract_classifier_factory

class knn_factory(abstract_classifier_factory):
    def __init__(self,k):
        self.k = k

    def train(self, data, labels):
        # TODO: takes train set.
        # TODO: return a knn_calsiffier.
        classifier = knn_classifier(data, labels, self.k)
        return classifier


class knn_classifier(abstract_classifier):
    def __init__(self, train_data, train_labels, k=1):
        # TODO initialize a knn classifier with a train set.
        self.data = train_data
        self.labels = train_labels
        self.num_samples = len(train_labels)
        self.k = k

    def classify(self, features):
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
    dist = None
    sum = 0
    for i in range(len(feature_list1)):
        x = feature_list1[i]
        y = feature_list2[i]
        sum += (x-y)**2
    dist = math.sqrt(sum)
    return dist


def evaluate(classifier_factory, k):
    AvgErr = None
    AvgAcc = None
    # k - number of folds.
    # TODO: run k-cross validation.
    # TODO: 1 set is chosen as test.
    # TODO: all others are chosen as train.
    # TODO: repeat until all k fold been as "test set".
    # TODO: return AvgError, AvgAccuracy
    return AvgErr, AvgAcc

