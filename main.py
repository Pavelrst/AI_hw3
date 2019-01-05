import numpy

from hw3_utils import load_data
from classifier import knn_factory

def split_crosscheck_groups(train_features, train_labels, num_folds):
    # TODO: split to num_folds.
    # TODO: write to file those folds:
    # file name: ecg_fold_i.data
    # use as an example ecg_examples.data.
    return None

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

    #folds = 2
    #split_crosscheck_groups(train_features, train_labels, folds)
    #knn3_fac = knn_factory(3)
    #acc, err = evaluate(knn3_fac, folds)

if __name__ == '__main__':
    main()
