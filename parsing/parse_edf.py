import os
import re
import glob
import joblib
import operator
import argparse
import subprocess

import numpy as np

import mne
# suppress warning
mne.set_log_level(verbose=False)


def parse_wfdb_description(desc_filename):
    """Parse the wfdb description files by finding the names of the
    channels used

    Parameters:
        desc_filename   - the input description file

    Returns:
        matches         - names of the channels
    """

    if os.path.exists(desc_filename):
        try:
            with open(desc_filename, "r") as f:
                regex = r".*Description: (.*)$"
                content = f.read()
                matches = re.findall(regex, content, flags=re.MULTILINE)
                return matches
        except:
            # handle exception
            raise


def parse_edf_with_mne(input_edf, desc_filename, channels, scale=1):
    """Parse the EDF file and extract the specified channels from the
    recording using the mne library

    Parameters:
        input_edf       - the input edf file, /path/to/edf
        desc_filename   - the input description file /path/to/description
        channels        - a list of channel names to be extracted
        scale           - int, scaling factor apply on the raw data values

    Returns:
        parsed_data     - (n, m) 2-D numpy array where n corresponds
                          to the number of channels
    """
    print("processing {} and {}".format(input_edf, desc_filename))

    assert len(channels) != 0, "length of channels must be greater than 0"

    parsed_data = []

    channels_in_edf = parse_wfdb_description(desc_filename)

    for channel in channels:
        if channel in channels_in_edf:
            channel_idx = channels_in_edf.index(channel)

            try:
                parsed_data.append(scale * mne.io.read_raw_edf(
                    input_edf, preload=True)[channel_idx][0][0])
            except:
                # handle exception
                raise

    return np.stack(parsed_data)


def parse_records_and_labels(record_names,
                             channels,
                             record_template,
                             label_template,
                             record_desc_template,
                             label_mapper,
                             label_mapper_to_int,
                             window_length,
                             sampling_frequency,
                             scale=1):
    """Parse all records and labels and return dictionary mapping
    record_name to (record, label) pair

    Note: This function makes the assumption that the head of the recording
    matches with the head of the hypnogram annotation. In case of mismatch
    between the ends, the longer one is truncated.

    Parameters:
        record_names        - list of strings, a list of record names
        channels            - list of strings, a list of channels
        record_template     - string, a string template for record (.edf)
        record_desc_template- string, a string template for record
                              description (.desc)
        label_template      - string, a string template for labels (.txt)
        label_mapper        - dictionary, mapping possible sleep stages to a
                              set of standardized sleep stages
        label_mapper_to_int - dictionary, mapping the set of standardized
                              sleep stages to integer
        window_length       - int, length of the window used in annotation
        sampling_frequency  - int, sampling frequency of the recording
        scale               - int, scaling factor applied on the raw data
                              values


    Returns:
        record_dictionary   - dictionary, a dictionary mapping
                                record_name to a (record, label) pair
                                record  - (n, m) 2-D numpy array of type float
                                label   - 1-D numpy array of type str
    """

    record_dictionary = {}

    for record_name in record_names:

        record_edf_filename = record_template.format(record_name)
        record_labels_filename = label_template.format(record_name)
        record_desc_filename = record_desc_template.format(record_name)

        record_edf = parse_edf_with_mne(record_edf_filename,
                                        record_desc_filename, channels, scale)
        record_labels = np.genfromtxt(record_labels_filename, dtype="str")

        record_labels = [label_mapper[label] for label in record_labels]
        record_labels = [label_mapper_to_int[label] for label in record_labels]

        recording_windows = int(
            record_edf.shape[1] / window_length / sampling_frequency)

        # heuristic to prune potentially problematic records
        if recording_windows < .5 * len(
                record_labels) or recording_windows > 2 * len(record_labels):
            print("Skip {}, length of recording: {}, length of hypnogram: {}".
                  format(record_name, recording_windows, len(record_labels)))
            continue

        # For sleep-edf, this happens with the SC* files
        if recording_windows < len(record_labels):
            record_labels = record_labels[:recording_windows]
        # For sleep-edf, this happens with the ST* files
        elif recording_windows > len(record_labels):
            record_edf = record_edf[:, :len(record_labels) * window_length *
                                    sampling_frequency]

        record_dictionary[record_name] = (record_edf,
                                          np.asarray(record_labels))

    return record_dictionary


def save_record_dictionary(record_dictionary, output_filename):
    """Save record dictionary to a pickle object

    Parameters:
        record_dictionary   - dictionary, mapping record_name to pair of
                              record data and record labels
        output_filename     - string, /path/to/output
    """

    with open(output_filename, "wb") as f:
        joblib.dump(record_dictionary, f)


def parse_edf_with_rdsamp(input_edf, desc_filename, channels):
    """Parse the EDF file and extract the specified channels from the
    recording using the rdsamp command from the wfdb library

    Parameters:
        input_edf       - the input edf file, /path/to/edf
        desc_filename   - the input description file /path/to/description
        channels        - a list of channel names to be extracted
    """

    print("processing {} and {}".format(input_edf, desc_filename))

    assert len(channels) != 0, "length of channels must be greater than 0"

    if input_edf[-4:] == ".edf":
        input_dir = os.path.dirname(input_edf)
        input_filename = os.path.basename(input_edf)

        arguments = "CWD=`pwd` && cd {} && rdsamp -c -p -H -v -r {} -s"

        output_filename = input_filename[:-4] + ".csv"

        channels_in_edf = parse_wfdb_description(desc_filename)

        for channel in channels:
            if channel in channels_in_edf:
                arguments += " {}".format(channels_in_edf.index(channel))

        arguments += " > {} && cd $CWD"

        arguments = arguments.format(input_dir, input_filename,
                                     output_filename)

        try:
            subprocess.run(
                arguments, shell=True, check=True, executable="/bin/bash")
        except:
            # TODO: handle exception
            raise


def parse_csv_records_and_labels(record_names,
                                 record_template,
                                 label_template,
                                 label_mapper=None):
    """Parse all csv records and labels and return dictionary mapping
    record_name to (record, label) pair

    Parameters:
        record_names        - list of strings, a list of record names
        record_template     - string, a string template for records (.edf)
        label_template      - string, a string template for labels (.txt)

    Returns:
        record_dictionary   - dictionary, a dictionary mapping
        record_name to a (record, label) pair
                                record  - 1-D numpy array of type float
                                label   - 1-D numpy array of type str
    """

    record_dictionary = {}

    for record_name in record_names:

        record_edf_filename = record_template.format(record_name)
        record_labels_filename = label_template.format(record_name)

        record_edf = np.loadtxt(
            record_edf_filename, skiprows=2, delimiter=",", usecols=1)
        record_labels = np.genfromtxt(record_labels_filename, dtype="str")

        if label_mapper is not None:
            record_labels = [label_mapper[label] for label in record_labels]

        record_dictionary[record_name] = (record_edf, record_labels)

    return record_dictionary


def summarize_channels(input_dir):
    """Get a summary about names of the channels used in the recordings in input_dir

    Parameters:
        input_dir   - the input directory containing the *.desc files
    """

    def update_dictionary(dic, matches):
        """Helper function to keep track of the number of occurrences of the channels
        """

        for match in matches:
            if match not in dic:
                dic[match] = 1
            else:
                dic[match] += 1
        return dic

    all_possible_channels = {}

    filenames = glob.glob(input_dir + "/*.desc")

    for filename in filenames:
        matches = parse_wfdb_description(filename)
        all_possible_channels = update_dictionary(all_possible_channels,
                                                  matches)

    all_possible_channels = sorted(
        all_possible_channels.items(),
        key=operator.itemgetter(1),
        reverse=True)

    print(all_possible_channels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="/path/to/input")
    ap.add_argument(
        "--channels",
        required=False,
        help=
        "list of channels (integers) to extract for the input edf file, separated by ,"
    )

    args = vars(ap.parse_args())


if __name__ == "__main__":
    main()
