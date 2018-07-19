import os
import joblib
import numpy as np

from .eeg_extract_features import extract_features
from .classification_formater import write_ucr_data_with_filename
from .transformation import down_sampling


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

    num_windows = int(
        parsed_record.shape[1] / sampling_frequency / window_length)

    assert num_windows == labels.shape[0], "labels do not match with data"

    num_samples_per_window = sampling_frequency * window_length

    slices = [
        slice(i * num_samples_per_window, (i + 1) * num_samples_per_window)
        for i in range(num_windows)
    ]

    # TODO: use np.split to avoid copying data
    windows = np.stack([
        np.stack([parsed_record[i, one_slice] for one_slice in slices])
        for i in range(parsed_record.shape[0])
    ])

    return windows


def perform_windowing_on_dict(record_dictionary,
                              sampling_frequency=256,
                              window_length=30):
    """Perform windowing operation on every record data in the record_dictionary

    Parameters:
        record_dictionary       - dictionary,
                                  key: record_name
                                  values: (record_data, record_labels)
                                  record_data: (n, m) numpy array
                                  record_labels: (w,) numpy array
        down_sampling_factor    - int, the down sampling factor to be applied
        window_length       - int, length of each window

    Returns:
        windowed_dictionary     - dictionary,
                                  key: record_name
                                  values: (record_data, record_labels)
                                  record_data: (n, w, m) numpy array
                                  record_labels: (w,) numpy array
    """

    windowed_dictionary = {}

    for key in record_dictionary:
        print("Processing: {}".format(key))

        data, labels = record_dictionary[key]

        try:
            window = (windowing(data, labels, sampling_frequency,
                                window_length), labels)

            windowed_dictionary[key] = window
        except AssertionError:
            print("[{}]: labels do not match with windowed data".format(key))
            continue

    return windowed_dictionary


def create_shapelets_dataset(record_dictionary,
                             channel,
                             channels,
                             down_sampling_factor=16):
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
        sorted_record_sep              - list of int, each element corresponds
                                  to the length of the labels of the corresponding
                                  record
    """

    channel_idx = channels.index(channel)

    sorted_records = sorted(list(record_dictionary.keys()))

    data = []
    labels = []
    sorted_record_sep = []

    for key in sorted_records:
        # (n, w, m)
        record_data, record_labels = record_dictionary[key]

        # (w, m/df)
        down_sampled_data = down_sampling(record_data[channel_idx],
                                          down_sampling_factor)

        extracted_idx = []

        for i in range(down_sampled_data.shape[0]):
            # in case all recordings are 0
            if down_sampled_data[i].sum() != 0:
                extracted_idx.append(i)

        extracted_data = down_sampled_data[extracted_idx]
        extracted_record_labels = record_labels[extracted_idx]

        data.append(extracted_data)
        labels.append(extracted_record_labels)
        sorted_record_sep.append(extracted_record_labels.shape[0])

    data = np.concatenate(data)
    labels = np.concatenate(labels)

    return data, labels, sorted_records, sorted_record_sep


def create_eeg_features_dataset(record_dictionary, channel, channels,
                                sampling_frequency, feature_dict):
    """Create EEG features dataset

    TODO: multiple channels

    Parameters:
        record_dictionary       - dictionary,
                                  key: record_name
                                  values: (record_data, record_labels)
                                  record_data: (n, w, m) numpy array
                                  record_labels: (w,) numpy array
        channel                 - string, the channel to use for shapelets
        channels                - list of string, the list of channels
                                  extracted from the edf recording
        sampling_frequency      - int, sampling frequency of the recording
        feature_dict            - dictionary,
                                  key: feature name
                                  value: feature parameters
    """

    channel_idx = channels.index(channel)

    sorted_records = sorted(list(record_dictionary.keys()))

    data = []
    labels = []
    ids_in_record = []
    sorted_record_sep = []

    for key in sorted_records:
        # (n, w, m)
        record_data, record_labels = record_dictionary[key]

        record_data_single_channel = record_data[channel_idx]

        extracted_data = []
        extracted_labels = []
        extracted_ids_in_record = []

        for i in range(record_data_single_channel.shape[0]):
            features = extract_features(record_data_single_channel[i],
                                        sampling_frequency, feature_dict)
            # if feature extraction succeeds, add windows
            """
            For 0906152-1, originally there are 1110 windows, after this preprocessing,
            there are 1107 windows
            """
            if np.isnan(features).sum() == 0:
                extracted_data.append(features)
                extracted_labels.append(i)
                extracted_ids_in_record.append((key, i))

        extracted_record_labels = record_labels[extracted_labels]

        data.append(np.stack(extracted_data))
        labels.append(extracted_record_labels)
        ids_in_record.append(extracted_ids_in_record)
        sorted_record_sep.append(extracted_record_labels.shape[0])

    data = np.concatenate(data)
    labels = np.concatenate(labels)
    ids_in_record = np.concatenate(ids_in_record)

    return data, labels, ids_in_record, sorted_records, sorted_record_sep


def save_shapelets_datasets_ucr(output_dir, prefix, data, labels,
                                sorted_records, sorted_record_sep):
    """Persist shapelets features dataset to hard disk

    Parameters:
        output_dir          - string, /path/to/output/dir
        prefix              - string, prefix to the output, e.g.,
                              TRAINING/TESTING
        data                - (n, m) 2-D numpy array of float
        labels              - (n,) 1-D numpy array of integer
        sorted_records      - list of string, a list of record names
                              where the order of the elements corresponds
                              to the data
        sorted_record_sep   - list of int, a list of record length where
                              the order of the elements corresponds to
                              the data
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ucr_data_template = "{}/{}"
    record_info_template = "{}/{}_record_info"

    write_ucr_data_with_filename(
        ucr_data_template.format(output_dir, prefix), labels, data)

    with open(record_info_template.format(output_dir, prefix), "wb") as f:
        joblib.dump((sorted_records, sorted_record_sep), f)
    f.close()


def save_eeg_features_datasets(output_dir, prefix, data, labels, ids_in_record,
                               sorted_records, sorted_record_sep):
    """Persist eeg features dataset to hard disk

    Parameters:
        output_dir          - string, /path/to/output/dir
        prefix              - string, prefix to the output, e.g.,
                              TRAINING/TESTING
        data                - (n, m) 2-D numpy array of float
        labels              - (n,) 1-D numpy array of integer
        ids_in_record       - (n, 2) numpy array of pair (string, int)
        sorted_records      - list of string, a list of record names
                              where the order of the elements corresponds
                              to the data
        sorted_record_sep   - list of int, a list of record length where
                              the order of the elements corresponds to
                              the data
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_template = "{}/{}_eeg_data"
    labels_template = "{}/{}_eeg_labels"
    ids_in_record_template = "{}/{}_eeg_ids_in_record"
    record_info_template = "{}/{}_eeg_record_info"

    np.savetxt(data_template.format(output_dir, prefix), data)
    np.savetxt(labels_template.format(output_dir, prefix), labels)
    np.savetxt(
        ids_in_record_template.format(output_dir, prefix),
        ids_in_record,
        fmt="%s")

    with open(record_info_template.format(output_dir, prefix), "wb") as f:
        joblib.dump((sorted_records, sorted_record_sep), f)
    f.close()
