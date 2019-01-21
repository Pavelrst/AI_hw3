import numpy as np
import math
import random
import pickle
from matplotlib import pyplot as plt
import csv

from hw3_utils import load_data
from hw3_utils import write_prediction
from classifier import knn_factory
from classifier import evaluate

# Used only in experiment.
from sklearn import tree
from sklearn.linear_model import Perceptron

# Competition model
from classifier import triple_model
from sklearn import preprocessing


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
    # Variables used for debug
    skip_knn = False
    skip_tree = False
    skip_perc = False

    train_features, train_labels, test_features = load_data('data/Data.pickle')

    # Split once the dataset to two folds.
    folds = 2
    #split_crosscheck_groups(train_features, train_labels, folds)

    if skip_knn != True:
        # Evaluating KNN with different k value:
        k_list = [1,3,5,7,13]
        acc_list = []
        err_list = []
        with open('experiments6.csv', mode='w', newline='') as csv_file:
            exp_writer = csv.writer(csv_file)
            for k in k_list:
                knn_fac = knn_factory(k)
                err, acc = evaluate(knn_fac, folds)
                print("k=",k," acc=",acc," err=",err)
                exp_writer.writerow([k, acc, err])
                acc_list.append(acc)
                err_list.append(err)

        # Plot KNN Results
        plt.subplot(2, 1, 1)
        plt.plot(k_list, acc_list, '--', color='g')
        plt.plot(k_list, acc_list, 'bo')
        plt.ylabel("Accuracy")
        plt.xlabel("k")
        plt.xticks(k_list)
        plt.subplot(2, 1, 2)
        plt.plot(k_list, err_list, '--', color='r')
        plt.plot(k_list, err_list, 'bo')
        plt.ylabel("Error")
        plt.xlabel("k")
        plt.xticks(k_list)
        plt.tight_layout()
        plt.show()

    # Perform classification for Perceptron and Tree and write to files.
    with open('experiments12.csv', mode='w', newline='') as csv_file:
        exp_writer = csv.writer(csv_file)
        if skip_tree != True:
            # Decision Tree experiment
            myTree = tree.DecisionTreeClassifier(criterion="entropy")
            err, acc = evaluate(myTree, folds)
            print("tree acc=",acc," tree err=",err)
            exp_writer.writerow([1, acc, err])

        if skip_perc != True:
            # Perceptron experiment
            myPerc = Perceptron(tol=1e-3, random_state=0)
            err, acc = evaluate(myPerc, folds)
            print("perceptron acc=", acc, " perceptron err=", err)
            exp_writer.writerow([2, acc, err])


    # Competition: Classify test_features
    my_model = triple_model()
    my_model.fit(train_features,train_labels)
    res = my_model.final_predict(preprocessing.scale(test_features))
    write_prediction(res)



if __name__ == '__main__':
    main()
