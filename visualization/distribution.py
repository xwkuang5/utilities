import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def vis_sleep_stages_distribution(records,
                                  filenames_template,
                                  label_mapper,
                                  sleep_stages=['W', '1', '2', '3', 'R', '?'],
                                  title="sleep stages distribution"):
    """Visualize sleep stages distribution from list of records with bar graph

    Parameters:
        records                 - list of records for visualization,
                                  e.g., ['0906152-1', '0907051-1']
        filenames_template      - template of the label filename,
                                  e.g., "/home/x4kuang/sleep_staging/
                                  sunnybrooks_datasets/{}.txt"
        label_mapper            - a dictionary mapping possible sleep
                                  stages to a standardized set of sleep
                                  stages, e.g., '1' -> '1', 'N1' -> '1'
        sleep_stages            - a list of sleep stages to be visualized
        title                   - title of the bar plot
    """

    sleep_stages_info = []

    for record in records:
        keep_sleep_stages_dict = dict(
            zip(sleep_stages, [0] * len(sleep_stages)))

        filename = filenames_template.format(record)

        labels = np.genfromtxt(filename, dtype="str")
        labels = [label_mapper[label] for label in labels]

        keys = set(sleep_stages)
        sorted_keys = sorted(sleep_stages)

        for label in labels:
            if label in keys:
                keep_sleep_stages_dict[label] += 1

        sleep_stages_info.append(
            [keep_sleep_stages_dict[label] for label in sorted_keys])

    sleep_stages_info = np.asarray(
        np.stack(sleep_stages_info), dtype=np.float64)

    sleep_stages_info_normed = np.divide(sleep_stages_info,
                                         np.sum(
                                             sleep_stages_info,
                                             axis=1,
                                             keepdims=True))
    sleep_stages_info_normed_mean = np.mean(sleep_stages_info_normed, axis=0)
    sleep_stages_info_normed_std = np.std(sleep_stages_info_normed, axis=0)

    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black']
    color = colors[len(sleep_stages) % 6]
    ind = np.arange(len(sleep_stages))
    plt.bar(
        ind,
        sleep_stages_info_normed_mean,
        color=color,
        yerr=sleep_stages_info_normed_std,
        align='center')
    plt.xticks(ind, sorted_keys)
    plt.title(title)
    plt.show()


def vis_sleep_stage(labels,
                    title,
                    label_mapper={
                        0: 'W',
                        1: '1',
                        2: '2',
                        3: '3',
                        4: 'R'
                    }):
    """Visualize sleep stages distribution from labels with bar graph

    Parameters:
        labels          - 1-D numpy array of int
        title           - string, title of the bar plot
        label_mapper    - dictionary, label mapper mapping integers to
                          sleep stages
    """

    color = "blue"
    keys = list(label_mapper.keys())
    values = [np.sum(labels == key) for key in keys]

    ind = np.arange(len(keys))

    plt.bar(ind, values, color=color, align='center')
    plt.xticks(ind, list(label_mapper.values()))
    plt.title(title)
    plt.show()


def show_PCA_plot(data, labels, label_mapper=None):

    data_centered = data - np.mean(data, axis=0, keepdims=True)
    embedded = PCA(n_components=2).fit_transform(data_centered)

    x_min, x_max = embedded[:, 0].min() - .5, embedded[:, 0].max() + .5
    y_min, y_max = embedded[:, 1].min() - .5, embedded[:, 1].max() + .5

    # Plot the training points
    plt.scatter(
        embedded[:, 0],
        embedded[:, 1],
        c=labels,
        cmap=plt.cm.Set1,
        edgecolor='k')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    seaborn_pair_plot(embedded, np.arange(2), labels, label_mapper)


def show_tSNE_plot(data, labels):
    """Show tSNE plot of the data

    TODO  add more tSNE parameters

    Parameters:
        data    : (n, m) numpy array
        labels  : (n,) numpy array
    """

    train_embedded = TSNE(n_components=2).fit_transform(data)

    x_min, x_max = train_embedded[:, 0].min() - .5, train_embedded[:, 0].max(
    ) + .5
    y_min, y_max = train_embedded[:, 1].min() - .5, train_embedded[:, 1].max(
    ) + .5

    # Plot the training points
    plt.scatter(
        train_embedded[:, 0],
        train_embedded[:, 1],
        c=labels,
        cmap=plt.cm.Set1,
        edgecolor='k')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def seaborn_pair_plot(data, features_names, labels, label_mapper=None):
    """Plot pairwise scatter plot with seaborn

    TODO add feature selection functions
         users should be able to give a list of index for which he/she
         would like to visualize.

    Parameters:
        data            : (n, m) 2-D numpy array
        features_names  : list, or (n,) 1-D array like of string
        labels          : list, or (n,) 1-D array like
        label_mapper    : dictionary, mapping label in labels to class
    """

    if label_mapper:
        labels_names = [label_mapper[label] for label in labels]
    else:
        labels_names = labels[:]

    data_frame = pd.DataFrame(
        data, index=np.arange(data.shape[0]), columns=features_names)
    data_frame = data_frame.assign(label=pd.Series(labels_names).values)

    sns.set(style="ticks", color_codes=True)

    _ = sns.pairplot(data_frame, hue="label")

    plt.show()

def seaborn_dist_plot(data, features_names):
    """Plot the distribution of the variables

    Parameters:
        data            : (n, m) 2-D numpy array
        features_names  : list, or (n,) 1-D array like of string
    """

    fig, axs = plt.subplots(ncols=len(features_names))

    for i in range(len(features_names)):
        sns.distplot(data[:, i], fit=norm, ax=axs[i])
        axs[i].set_title(features_names[i])
     
    fig.set_figwidth(10*len(features_names))
    fig.set_figheight(20)

    plt.tight_layout()
    plt.show()
