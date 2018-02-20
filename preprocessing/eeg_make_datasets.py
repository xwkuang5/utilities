import pickle
import numpy as np

from classification_formater import write_ucr_data_with_filename

def down_sampling(record, down_sampling_factor=16):
    """Perform down sampling on the input record

    Parameters:
        record                  - 1-D numpy array of type float
                                  In the case of 2-D array, down
                                  sampling is applied on the second axis (axis=1)
        down_sampling_factor    - int, the down sampling factor to be applied

    Returns:
        record_down_sampled     - A numpy array of type float, down sampled version
    """

    if len(record.shape) == 1:
        return record[slice(0, record.shape[0], down_sampling_factor)]
    else:
        row_idx = np.arange(record.shape[0])
        col_idx = np.arange(0, record.shape[1], down_sampling_factor)

        return record[np.ix_(row_idx, col_idx)]

def windowing(parsed_record, labels, sampling_frequency=256, window_length=30):
    """Split the input record into multiple windows of length window_length

    Parameters:
        parsed_record       - (n, m) 2-D numpy array of type float
                              where n corresponds to the number of channels
        labels              - 1-D numpy array
        sampling_frequency  - int, sampling frequency of the recording
        window_length       - int, length of each window

    Returns:
        windows              - (n, w, m) numpy array of dtype float
                               where n corresponds to the number of channels
                               w corresponds to the number of windows
                               m corresponds to the number of samples
    """

    num_windows = int(parsed_record.shape[1] / sampling_frequency / window_length)

    assert num_windows == labels.shape[0], "labels do not match with data"

    num_samples_per_window = sampling_frequency * window_length

    slices = [slice(i*num_samples_per_window, (i+1)*num_samples_per_window) for i in range(num_windows)]

    # TODO: use np.split to avoid copying data
    windows = np.stack([np.stack([parsed_record[i, one_slice] for one_slice in slices]) for i in range(parsed_record.shape[0])])

    return windows

def perform_windowing_on_dict(record_dictionary, sampling_frequency=256, window_length=30):

    windowed_dictionary = {}

    for key in record_dictionary:
        data, labels = record_dictionary[key]

        windowed_dictionary[key] = (windowing(data, labels, sampling_frequency, window_length), labels)

    return windowed_dictionary

def train_test_split(record_dictionary, ratio=.5, shuffle=True):

    num_training_records = int(len(record_dictionary) * ratio)

    keys = list(record_dictionary.keys())

    training_records = np.random.choice(keys, num_training_records, replace=False)
    testing_records = [key for key in keys if key not in training_records]

    if shuffle:
        np.random.shuffle(training_records)
        np.random.shuffle(testing_records)

    training_dictionary = {record: record_dictionary[record] for record in training_records}
    testing_dictionary = {record: record_dictionary[record] for record in testing_records}

    return training_dictionary, testing_dictionary

def create_shapelets_datasets(record_dictionary, channel, channels, down_sampling_factor=16):
    """Create datasets for shapelets algorithm

    Parameters:
        record_dictionary       - dictionary,
                                  key: record_name
                                  values: (record_data, record_labels)
                                  record_data: (n, w, m) numpy array
                                  record_labels: (w,) numpy array
        channel                 - string, the channel to use for shapelets
        channels                - list of string, the list of channels
                                  extracted from the edf recording
        down_sampling_factor    - int, the down sampling factor to be applied

    Returns:
        data                    - (W, m) numpy array of float
        labels                  - (W,) numpy array of int
        sorted_records          - list of string, each element corresponds
                                  to a record used in this dataset
        record_sep              - list of int, each element corresponds
                                  to the length of the labels of the corresponding
                                  record
    """

    channel_idx = channels.index(channel)

    sorted_records = sorted(list(record_dictionary.keys()))

    data = []
    labels = []
    record_sep = []

    for key in sorted_records:
        # (n, w, m)
        record_data, record_labels = record_dictionary[key]

        # (w, m/df)
        down_sampled_data = down_sampling(record_data[channel_idx], down_sampling_factor)

        data.append(down_sampled_data)
        labels.append(record_labels)
        record_sep.append(record_labels.shape[0])

    data = np.concatenate(data)
    labels = np.concatenate(labels)

    return data, labels, sorted_records, record_sep

def save_shapelets_datasets_ucr(output_dir, prefix, data, labels, records, record_sep):

    ucr_data_template = "{}/{}_shapelets"
    record_info_template = "{}/{}_shapelets_record_info"

    write_ucr_data_with_filename(ucr_data_template.format(output_dir, prefix), labels, data)

    with open(record_info_template.format(output_dir, prefix), "wb") as f:
        pickle.dump((records, record_sep), f)
    f.close()
