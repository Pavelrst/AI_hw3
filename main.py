import numpy as np
import math
import random
from matplotlib import pyplot as plt

from hw3_utils import load_data
from classifier import knn_factory
from classifier import evaluate

def split_crosscheck_groups(train_features, train_labels, num_folds):
    # TODO: split to num_folds.
    # TODO: write to file those folds:
    # file name: ecg_fold_i.data
    # use as an example ecg_examples.data.
    folds_idx = []
    temp_idx = range(num_folds)
    rep = math.ceil(len(train_labels) / num_folds)
    for r in range(rep):
        folds_idx+=temp_idx
    folds_idx = folds_idx[:len(train_labels)]
    random.shuffle(folds_idx)

    features_folds = [[] for x in range(num_folds)]
    labels_folds = [[] for x in range(num_folds)]
    for i in range(len(train_labels)):
        curr_feature = train_features[i]
        curr_lable = train_labels[i]
        curr_fold = folds_idx[i]
        features_folds[curr_fold].append(curr_feature)
        labels_folds[curr_fold].append(curr_lable)
    #print("label folds", labels_folds)
    #print("features_folds", features_folds)

    # TODO: save it to file
    return


def evaluate(classifier_factory, k):
    # k - number of folds.
    # TODO: run k-cross validation.
    # TODO: 1 set is chosen as test.
    # TODO: all others are chosen as train.
    # TODO: repeat until all k fold been as "test set".
    # TODO: return AvgError, AvgAccuracy
    return None

def main():
    print("main")

    train_features, train_labels, test_features = load_data('data/Data.pickle')
    print("features:",train_features.shape)
    print("labels:",train_labels)

    folds = 2
    split_crosscheck_groups(train_features, train_labels, folds)

    # Experiment:
    k_list = [1,3,5,7,13]
    acc_list = []
    err_list = []
    for k in k_list:
        knn3_fac = knn_factory(3)
        acc, err = evaluate(knn3_fac, folds)
        acc_list.append(acc)
        err_list.append(err)

    plt.subplot(2, 1, 1)
    plt.plot(k_list, acc_list)
    plt.subplot(2, 1, 2)
    plt.plot(k_list, err_list)
    plt.show()




if __name__ == '__main__':
    main()
