import os
import pickle
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

import copy


def compute_sample_weight(labels, mode="uniform"):
    if mode == "uniform":
        return [1] * labels.shape[0]
    elif mode == "proportional":
        bincount = np.bincount(labels) / labels.shape[0]
        return np.asarray(
            [bincount[labels[i]] for i in range(labels.shape[0])],
            dtype=np.float64)
    elif mode == "reverse-proportional":
        bincount = labels.shape[0] / np.bincount(labels)
        return np.asarray(
            [bincount[labels[i]] for i in range(labels.shape[0])],
            dtype=np.float64)


def run_classifier(train_data,
                   train_labels,
                   test_data,
                   test_labels,
                   name,
                   classifier,
                   parameters,
                   num_runs=5):
    """Run a classifier multiple times

    Parameters:
        train_data      : (n, m) numpy array
        train_labels    : (n,) numpy array
        test_data       : (n, m) numpy array
        test_labels     : (n,) numpy array
        name            : string, name of the classifier
        classifier      : classifer implementing the fit and predict method
        parameters      : list key-value pairs, parameters to be used for grid search
        n_runs          : int, number of runs to repeat for a classifer
    """

    classifiers = []
    kappas = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for run in range(num_runs):

        clf = GridSearchCV(copy.deepcopy(classifier), parameters, cv=5)

        if name == "GaussianNB":
            clf.fit(
                train_data,
                train_labels,
                sample_weight=compute_sample_weight(train_labels,
                                                    "proportional"))
        else:
            clf.fit(train_data, train_labels)

        test_predictions = clf.predict(test_data)

        kappas.append(cohen_kappa_score(test_labels, test_predictions))
        accuracies.append(accuracy_score(test_labels, test_predictions))
        f1s.append(f1_score(test_labels, test_predictions, average="weighted"))
        precisions.append(
            precision_score(test_labels, test_predictions, average="weighted"))
        recalls.append(
            recall_score(test_labels, test_predictions, average="weighted"))

        classifiers.append(clf)

    best_idx = np.argmax(accuracies)

    best_clf = classifiers[best_idx]

    test_predict_labels = best_clf.predict(test_data)

    return {
        "kappa": kappas,
        "accuracy": accuracies,
        "precision": precisions,
        "recall": recalls,
        "f1": f1s,
        "clf": best_clf,
        "predictions": test_predict_labels
    }


def run_experiments(input_dir,
                    n_runs,
                    names,
                    classifiers,
                    parameters,
                    output_dir,
                    balanced=False):
    """
    Parameters:
        input_dir       : string, /path/to/input/directory
        n_runs          : int, number of runs to repeat for a classifer
        names           : list of string, names of the classifiers
        classifiers     : list of classifer implementing the fit and predict method
        parameters      : dictionary, key: classifier name
                                      value: list key-value pairs, parameters to be used for grid search
        output_dir      : string, /path/to/output/directory
        balanced        : boolean, whether to use balanced training dataset
    """

    train_data_dir = "/".join([input_dir, "train_eeg_data"])
    train_labels_dir = "/".join([input_dir, "train_eeg_labels"])
    train_record_info = "/".join([input_dir, "train_eeg_record_info"])

    balanced_train_data_dir = "/".join([input_dir, "balanced_train_eeg_data"])
    balanced_train_labels_dir = "/".join(
        [input_dir, "balanced_train_eeg_labels"])

    test_data_dir = "/".join([input_dir, "test_eeg_data"])
    test_labels_dir = "/".join([input_dir, "test_eeg_labels"])
    test_record_info = "/".join([input_dir, "test_eeg_record_info"])

    train_data = np.loadtxt(train_data_dir)
    train_labels = np.loadtxt(train_labels_dir, dtype=np.uint8)

    with open(train_record_info, "rb") as f:
        (train_records, train_records_sep) = pickle.load(f)
    f.close()

    if balanced:
        balanced_train_data = np.loadtxt(balanced_train_data_dir)
        balanced_train_labels = np.loadtxt(
            balanced_train_labels_dir, dtype=np.uint8)

    test_data = np.loadtxt(test_data_dir)
    test_labels = np.loadtxt(test_labels_dir, dtype=np.uint8)

    with open(test_record_info, "rb") as f:
        (test_records, test_records_sep) = pickle.load(f)
    f.close()

    if not os.path.exists(output_dir):
        os.path.mkdir(output_dir)

    results_dictionary = {}

    for idx, clf in enumerate(classifiers):

        if balanced:
            results_dictionary[names[idx]] = run_classifier(
                balanced_train_data, balanced_train_labels, test_data,
                test_labels, names[idx], clf, parameters[idx], n_runs)
        else:
            results_dictionary[names[idx]] = run_classifier(
                train_data, train_labels, test_data, test_labels, names[idx],
                clf, parameters[idx], n_runs)

    with open("/".join([output_dir, "results_dictionary"]), "wb") as f:
        pickle.dump(results_dictionary, f)
    f.close()

    return results_dictionary


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--input_dir", required=True, help="/path/to/input")
    ap.add_argument("--output_dir", required=True, help="/path/to/output")
    ap.add_argument("--num_runs", required=True, help="number of runs")

    args = vars(ap.parse_args())

    classifiers = [
        GaussianNB(),
        KNeighborsClassifier(),
        svm.SVC(max_iter=1000),
        LogisticRegression(solver="lbfgs"),
        MLPClassifier(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        DecisionTreeClassifier()
    ]

    names = [
        "GaussianNB", "KNearestNeighbor", "SupportVectorMachine",
        "LogisticRegression", "Perceptron", "RandomForest", "GradientBoosting",
        "DecisionTree"
    ]

    parameters = [{}, {
        "n_neighbors": [5, 10, 30, 50],
        "weights": ["uniform", "distance"]
    }, {
        "kernel": ["linear", "rbf", "poly"],
        "class_weight": ["balanced", None]
    }, {
        "multi_class": ["ovr", "multinomial"],
        "class_weight": ["balanced", None]
    }, {
        "hidden_layer_sizes": [50, 100, 200]
    }, {
        "criterion": ["gini", "entropy"],
        "class_weight": ["balanced", None]
    }, {
        "max_depth": [3, 4, 5]
    }, {
        "criterion": ["gini", "entropy"],
        "class_weight": ["balanced", None]
    }]

    _ = run_experiments(args["input_dir"], int(args["num_runs"]), names,
                        classifiers, parameters, args["output_dir"])


if __name__ == "__main__":
    main()
