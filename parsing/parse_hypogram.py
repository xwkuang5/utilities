import glob
import numpy as np
import pandas as pd


def parse_hypnogram(directory, hypnogram_suffix, output_suffix, window_length):
    """Parse hypnograms under the specified directory

    Parameters:
        directory           : string, /path/to/directory
        hypnogram_suffix    : string, suffix of the hypnogram files
                              e.g., hyp for sleep-edf dataset
        output_suffix       : string, suffix of the output sleep stage
                              files
        window_length       : int, length of the window used for sleep
                              staging
    """

    filenames = glob.glob("/".join([directory,
                                    "*{}".format(hypnogram_suffix)]))

    for filename in filenames:

        try:
            hypnogram = parse_hyp_txt_from_rdann(filename, window_length)

            output_filename = filename[:-len(hypnogram_suffix)] + output_suffix
            np.savetxt(output_filename, hypnogram, fmt="%s")

            print("parse hypnogram [{}] -> [{}]".format(
                filename, output_filename))
        except:
            # TODO: handle exception
            raise


def parse_hyp_txt_from_rdann(filename, window_length):
    """Parse hypnogram file returned from rdann command

    file.hypnogram_suffix is assumed to be the output of the command
        rdann -r record -a hypnogram_suffix

    Parameters:
        filename                : string, /path/to/input
        window_length           : int, length of the window used for sleep
                              staging

    Returns:
        unfolded_sleep_stages   : list of string, sleep stages
                                  corresponding to the hypnogram
    """

    data_frame = pd.read_csv(filename, sep="\s+", header=None, engine="python")

    ndarray = data_frame.values

    sleep_stages = ndarray[:, 7]

    duration = ndarray[:, 9]

    sleep_stages = [sleep_stage[-1] for sleep_stage in sleep_stages]

    unfolded_sleep_stages = [
        sleep_stage for (idx, val) in enumerate(duration)
        for sleep_stage in sleep_stages[idx] * (val // window_length)
    ]

    return unfolded_sleep_stages


parse_hypnogram("/home/x4kuang/sleep_staging/sleep_edf_datasets", "csv", "txt",
                30)
