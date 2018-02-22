import numpy as np
import matplotlib.pyplot as plt

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black']


def vis_sleep_stages_distribution(records,
                                  filenames_template,
                                  label_mapper,
                                  sleep_stages=['W', '1', '2', '3', 'R', '?'],
                                  title="sleep stages distribution"):
    """Visualize sleep stages distribution with bar graph

    Parameters:
        records                 - list of records for visualization, e.g., ['0906152-1', '0907051-1']
        filenames_template      - template of the label filename, e.g., "/home/x4kuang/sleep_staging/sunnybrooks_datasets/{}.txt"
        label_mapper            - a dictionary mapping possible sleep stages to a standardized set of sleep stages, e.g., '1' -> '1', 'N1' -> '1'
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

def vis_sleep_stage(labels, title, label_mapper={0: 'W', \
                                                  1: '1', \
                                                  2: '2', \
                                                  3: '3', \
                                                  4: 'R'}):
    """Visualize sleep stages distribution with bar graph

    Parameters:
        labels          - 1-D numpy array of int
        title           - string, title of the bar plot
        label_mapper    - dictionary, label mapper mapping integers to sleep stages 
    """

    color = "blue"
    keys = list(label_mapper.keys())
    values = [np.sum(labels == key) for key in keys]
    
    ind = np.arange(len(keys))
    
    plt.bar(
        ind,
        values,
        color=color,
        align='center')
    plt.xticks(ind, list(label_mapper.values()))
    plt.title(title)
    plt.show()