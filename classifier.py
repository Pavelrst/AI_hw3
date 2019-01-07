import math
import random
import numpy as np
from hw3_utils import abstract_classifier
from hw3_utils import abstract_classifier_factory
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

class mlp_concurrent_model():
    def __init__(self,min_acc=0.96,max_iter=5,max_acc=0.97):
        self.min_acc = min_acc
        self.max_iter = max_iter
        self.max_acc = max_acc

    def fit(self,train_features, train_labels):
        # TODO: train 3 different classifiers
        alph = 1e-1
        lr = 'constant'
        layers = (100, 200, 200,50)
        train_data_mlp = train_features[:750]
        train_labels_mlp = train_labels[:750]
        self.val_data_mlp = train_features[-250:]
        self.val_labels_mlp = train_labels[-250:]

        acc = 0
        iter = 0
        print("training first net")
        while acc < self.min_acc and iter < self.max_iter:
            iter += 1
            self.myMLP1 = MLPClassifier(tol=1e-5, shuffle=True, max_iter=200, learning_rate=lr, activation='relu',
                                        solver='lbfgs', alpha=alph,hidden_layer_sizes=(50,100,20))
            self.myMLP1.fit(train_data_mlp, train_labels_mlp)
            res1 = self.myMLP1.predict(self.val_data_mlp)
            acc = self.calc_acc_err(res1,self.val_labels_mlp)
            if acc > self.max_acc: break

        acc = 0
        iter = 0
        print("training second net")
        while acc < self.min_acc and iter < self.max_iter:
            iter += 1
            self.myMLP2 = MLPClassifier(tol=1e-3, shuffle=True, max_iter=200, learning_rate=lr, activation='relu',
                                        solver='adam', alpha=alph, hidden_layer_sizes=(20,100,100,20))
            self.myMLP2.fit(train_data_mlp, train_labels_mlp)
            res2 = self.myMLP2.predict(self.val_data_mlp)
            acc = self.calc_acc_err(res2, self.val_labels_mlp)
            if acc > self.max_acc: break

        acc = 0
        iter = 0
        print("training third net")
        while acc < self.min_acc and iter < self.max_iter:
            iter +=1
            self.myMLP3 = MLPClassifier(tol=1e-5, shuffle=True, max_iter=200, learning_rate=lr, activation='relu',
                                        solver='lbfgs', alpha=alph, hidden_layer_sizes=(50,200,5))
            self.myMLP3.fit(train_data_mlp, train_labels_mlp)
            res3 = self.myMLP3.predict(self.val_data_mlp)
            acc = self.calc_acc_err(res3, self.val_labels_mlp)
            if acc > self.max_acc: break

        temp_res = np.zeros(len(res1))
        for i in range(len(res1)):
            temp_res[i]=int(res1[i])+int(res2[i])+int(res3[i])

        final_res = np.zeros_like(res1)
        for i in range(len(res1)):
            if temp_res[i] <2:
                final_res[i] = False
            else:
                final_res[i] = True
        print("final validation result =>>>")
        self.calc_acc_err(final_res,self.val_labels_mlp)

    def calc_acc_err(self,res,val_labels_mlp):
        right = 0
        wrong = 0
        for i in range(len(res)):
            if res[i] == val_labels_mlp[i]:
                right += 1
            else:
                #print("wrong index = ",i)
                wrong += 1
        acc = right / len(res)
        err = wrong / len(res)
        print("mlp acc=", acc, " err=", err)
        return acc

    def final_predict(self,test_features):
        return None

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
        #print("classify called ", self.calls)
        self.calls += 1
        # pass on all data and generate dist for all.
        debug = True
        if (debug == False):
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
        else:
            # Random answer for debug usage
            return random.choice((True, False))


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

        classifications = []
        if isinstance(classifier_factory, knn_factory):
            print("we got knn")
            classifier = classifier_factory.train(train_data, train_labels)
            for sample in test_data:
                classifications.append(classifier.classify(sample))
        elif isinstance(classifier_factory, tree.DecisionTreeClassifier):
            print("we got tree")
            classifier_factory = classifier_factory.fit(np.array(train_data), train_labels)
            classifications = classifier_factory.predict(np.array(test_data))
        elif isinstance(classifier_factory, Perceptron):
            print("we got preceptron")
            classifier_factory = classifier_factory.fit(np.array(train_data), train_labels)
            classifications = classifier_factory.predict(np.array(test_data))
        elif isinstance(classifier_factory, MLPClassifier):
            #print("we gor MLP")
            #train_data = train_data / np.linalg.norm(train_data)
            classifier_factory = classifier_factory.fit(np.array(train_data), train_labels)
            #test_data = test_data / np.linalg.norm(test_data)
            classifications = classifier_factory.predict(np.array(test_data))

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

