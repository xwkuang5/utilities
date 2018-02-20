import os
import re
import glob
import operator
import argparse
import subprocess

import numpy as np

def parse_wfdb_description(desc_filename):
    """Parse the wfdb description files by finding the names of the channels used

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
            f.close()
        except:
            # TODO: handle exception
            raise

def parse_edf(input_edf, desc_filename, channels, sampling_frequency=256):
    """Parse the EDF file and extract the specified channels from the recording

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

        arguments = arguments.format(input_dir, input_filename, output_filename)

        try:
            subprocess.run(arguments, shell=True, check=True, executable="/bin/bash")
        except:
            # TODO: handle exception
            raise

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
        all_possible_channels = update_dictionary(all_possible_channels, matches)

    all_possible_channels = sorted(all_possible_channels.items(), key=operator.itemgetter(1), reverse=True)

    print(all_possible_channels)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="/path/to/input")
    ap.add_argument("--channels", required=False, help="list of channels (integers) to extract for the input edf file, separated by ,")

    args = vars(ap.parse_args())

    if args["channels"] != None:
        channels = args["channels"].split(",")

if __name__ == "__main__":
    main()
