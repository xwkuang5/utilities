import itertools
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title="Confusion matrix",
                          cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Note: directly copied from sklearn tutorial
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def metric_barplot(title, metric_name, x_axis_names, data):
    """Plot bar graph with seaborn

    Parameters:
        title       : string, title of the plot
        metric_name : string, name of the metric
        x_axis_names: list of strings, values of the x-axis
        data        : (n, m) array, n: number of values on the axis
                                    m: number of values for each value on the x-axis
    """

    data_frame = pd.DataFrame(
        data=np.transpose(data), columns=pd.Series(x_axis_names))

    sns.barplot(data=data_frame, capsize=.2)

    plt.title(title)
    plt.xticks(np.arange(len(x_axis_names)), x_axis_names, rotation=45)


def compute_scores(labels,
                   predictions,
                   average=None,
                   classes=None,
                   title="Metrics",
                   verbose=True):
    """Plot f1, precision, recall in matrix form

    TODO look into roc_auc_score for multiclass

    Parameters:
        labels          : (n,) 1-D numpy array, ground truth labels
        predictions     : (n,) 1-D numpy array, predicted labels
        average         : string, f1, precision, recall score parameter
        classes         : list of string, names of the classes
        title           : string, title of the plot
        verbose         : boolean, if true, print accuracy, f1,
                          precision, recall score
    """

    scores = {}
    scores["kappa"] = cohen_kappa_score(labels, predictions)
    scores["accuracy"] = accuracy_score(labels, predictions)
    scores["f1"] = f1_score(labels, predictions, average=average)
    scores["precision"] = precision_score(labels, predictions, average=average)
    scores["recall"] = recall_score(labels, predictions, average=average)

    if verbose:
        print("{:<10}: {}".format("kappa", scores["kappa"]))
        print("{:<10}: {}".format("accuracy", scores["accuracy"]))
        print("{:<10}: {}".format("f1", scores["f1"]))
        print("{:<10}: {}".format("precision", scores["precision"]))
        print("{:<10}: {}".format("recall", scores["recall"]))

        if average == None:

            matrix = np.stack(
                [scores["f1"], scores["precision"], scores["recall"]])

            plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(title)
            plt.colorbar()

            x_tick_marks = np.arange(len(classes))
            y_tick_marks = np.arange(3)

            plt.xticks(x_tick_marks, classes, rotation=45)
            plt.yticks(y_tick_marks, ["f1", "precision", "recall"])

            fmt = '.2f'
            thresh = matrix.max() / 2.
            for i, j in itertools.product(
                    range(matrix.shape[0]), range(matrix.shape[1])):
                plt.text(
                    j,
                    i,
                    format(matrix[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if matrix[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('Metrics')
            plt.xlabel('Sleep stages')

            plt.show()

    return scores
