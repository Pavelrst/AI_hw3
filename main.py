import numpy as np
import math
import random
import pickle
from matplotlib import pyplot as plt

from hw3_utils import load_data
from classifier import knn_factory
from classifier import evaluate
from classifier import load_k_fold_data


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

    #print("labels from fold1     =",labels_folds[1])
    #print("features from fold1[0]  =", features_folds[1][0])

    # save it to file in some readable format.
    for fold_idx in range(len(labels_folds)):
        filename = "ecg_fold_" + str(fold_idx) + ".data"
        file = open(filename, "w")
        for i in range(len(labels_folds[fold_idx])):
            for f in range(len(features_folds[fold_idx][i])):
                file.write(str(features_folds[fold_idx][i][f]))
                file.write("\n")
            file.write(str(labels_folds[fold_idx][i]))
            file.write("\n")

        file.close()

    return




def main():
    print("main")

    train_features, train_labels, test_features = load_data('data/Data.pickle')
    #print("features:",train_features.shape)
    #print("labels:",train_labels)

    folds = 2
    split_crosscheck_groups(train_features, train_labels, folds)

    # Experiment:
    #k_list = [1,3,5,7,13]
    k_list = [1]
    acc_list = []
    err_list = []
    for k in k_list:
        knn3_fac = knn_factory(3)
        acc, err = evaluate(knn3_fac, folds)
        acc_list.append(acc)
        err_list.append(err)
#
    plt.subplot(2, 1, 1)
    plt.plot(k_list, acc_list)
    plt.subplot(2, 1, 2)
    plt.plot(k_list, err_list)
    plt.show()




if __name__ == '__main__':
    main()
