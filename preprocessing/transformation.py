import numpy as np

def down_sampling(record, down_sampling_factor=16):
    """Perform down sampling on the input record

    Parameters:
        record                  - 1-D numpy array of type float
                                  In the case of 2-D array, down
                                  sampling is applied on the second axis
                                  (axis=1)
        down_sampling_factor    - int, the down sampling factor to be
                                  applied

    Returns:
        record_down_sampled     - A numpy array of type float, down
                                  sampled version
    """

    if len(record.shape) == 1:
        return record[slice(0, record.shape[0], down_sampling_factor)]
    else:
        row_idx = np.arange(record.shape[0])
        col_idx = np.arange(0, record.shape[1], down_sampling_factor)

        return record[np.ix_(row_idx, col_idx)]

def remove_labels(record_dictionary, labels_to_remove=[5], inplace=True):
    """Remove the labels that are listed

    Parameters:
        record_dictionary       - dictionary,
                                  key: record_name
                                  values: (record_data, record_labels)
                                  record_data: (n, w, m) numpy array
                                  record_labels: (w,) numpy array
        labels_to_remove        - list of int, a list of labels to be removed

    Returns:
        purged_dictionary       - dictionary,
                                  key: record_name
                                  values: (record_data, record_labels)
                                  record_data: (n, w, m) numpy array
                                  record_labels: (w,) numpy array
    """

    if not inplace:
        purged_dictionary = {}

    labels_to_remove_set = set(labels_to_remove)

    for key in record_dictionary:

        data, labels = record_dictionary[key]

        remove_ids = [idx for (idx, val) in enumerate(labels) if val in labels_to_remove_set]

        if inplace:
            record_dictionary[key] = (np.delete(data, remove_ids, 1), np.delete(labels, remove_ids, 0))
        else:
            purged_dictionary[key] = (np.delete(data, remove_ids, 1), np.delete(labels, remove_ids, 0))

    if not inplace:
        return purged_dictionary

def balance_classes(data, labels):
    """Balance class distribution in the data

    Note call this function will destroy row structure of the data
    In case of EEG records, record ordering will be lost

    Parameters:
        data            - (n, m) numpy array of float
        labels          - (n,) numpy array of int

    Returns:
        balanced_data   - (n, m) numpy array of float
        balanced_labels - (n,)  numpy array of int
    """

    index_dict = {}

    for idx, label in enumerate(labels):
        if label not in index_dict:
            index_dict[label] = [idx]
        else:
            index_dict[label] += [idx]

    index_list = list(index_dict.values())

    min_balanced_number = min([len(l) for l in index_list])

    index_to_take_list = np.concatenate([np.random.choice(l, min_balanced_number, replace=False) for l in index_list])

    np.random.shuffle(index_to_take_list)

    return data[index_to_take_list], labels[index_to_take_list]


def train_test_split(record_dictionary, ratio=.5):
    """Split the record_dictionary into training set and testing set

    Parameters:
        record_dictionary       - dictionary,
                                  key: record_name
                                  values: (record_data, record_labels)
        ratio                   - float, train - test ratio

    Returns:
        training_dictionary     - dictionary,
                                  key: record_name
                                  values: (record_data, record_labels)
        testing_dictionary      - dictionary,
                                  key: record_name
                                  values: (record_data, record_labels)
    """

    num_training_records = int(len(record_dictionary) * ratio)

    keys = list(record_dictionary.keys())

    training_records = np.random.choice(
        keys, num_training_records, replace=False)
    testing_records = [key for key in keys if key not in training_records]

    training_dictionary = {
        record: record_dictionary[record]
        for record in training_records
    }
    testing_dictionary = {
        record: record_dictionary[record]
        for record in testing_records
    }

    return training_dictionary, testing_dictionary
