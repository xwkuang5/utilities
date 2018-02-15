import glob
import operator
import numpy as np
import argparse

from parse_wfdb_description import parse_wfdb_description

def parse_edf(input_edf, channels=None):
    pass

def get_channel_names_summary(input_dir):
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
