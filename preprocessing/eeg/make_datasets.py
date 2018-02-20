import numpy as np

def down_sampling(record, down_sampling_factor=16):
    """Perform down sampling on the input record

    Parameters:
        record                  - 1-D numpy array of type float
                                  In the case of 2-D array, down sampling is applied on the second axis (axis=1)
        down_sampling_factor    - int, the down sampling factor to be applied

    Returns:
        record_down_sampled     - 1-D numpy array of type float, down sampled version
    """

    return record[slice(0, record.shape[0], down_sampling_factor)]

def windowing(parsed_record, labels, sampling_frequency=256, window_length=30):
    """Split the input record into multiple windows of length window_length

    Parameters:
        parsed_record       - 1-D numpy array of type float
        labels              - 1-D numpy array
        sampling_frequency  - int, sampling frequency of the recording
        window_length       - int, length of each window

    Returns:
        windows              - (m, n) numpy array of dtype float
    """

    num_windows = int(parsed_record.shape[0] / sampling_frequency / window_length)

    assert num_windows == labels.shape[0], "annotations do not match with data"

    num_samples_per_window = sampling_frequency * window_length

    slices = [slice(i*num_samples_per_window, (i+1)*num_samples_per_window) for i in range(num_windows)]

    windows = np.stack([parsed_record[one_slice] for one_slice in slices])

    return windows

def parse_csv_records(record_names, record_template, label_template):
    """Parse all csv records

    Parameters:
        record_names        - list of strings, a list of record names
        record_template     - string, a string template for records (.edf)
        label_template      - string, a string template for labels (.txt)

    Returns:
        record_dictionary   - dictionary, a dictionary mapping record_name to a (record, label) pair
                                record  - 1-D numpy array of type float
                                label   - 1-D numpy array of type str
    """

    record_dictionary = {}

    for record_name in record_names:

        record_edf_filename = record_template.format(record_name)
        record_labels_filename = label_template.format(record_name)

        record_edf = np.loadtxt(record_edf_filename, skiprows=2, delimiter=",", usecols=1)
        record_labels = np.genfromtxt(record_labels_filename, dtype="str")

        record_dictionary[record_name] = (record_edf, record_labels)

    return record_dictionary
