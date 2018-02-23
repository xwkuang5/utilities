import glob
import numpy as np
import pandas as pd

def parse_hypnogram(directory, hypogram_suffix, output_suffix, window_length):

    filenames = glob.glob("/".join([directory, "*{}".format(hypogram_suffix)]))

    for filename in filenames:

        try:
            hypnogram = parse_hyp_txt_from_rdedfann(filename, window_length)

            output_filename = filename[:-len(hypogram_suffix)] + output_suffix
            np.savetxt(output_filename, hypnogram, fmt="%s")
        except:
            # TODO: handle exception
            raise


def parse_hyp_txt_from_rdedfann(filename, window_length):

    data_frame = pd.read_csv(filename, sep="\s+", header=None)

    ndarray = data_frame.values

    sleep_stages = ndarray[:, 2]

    duration = ndarray[:, 7]

    sleep_stages = [sleep_stage[-1] for sleep_stage in sleep_stages]

    unfolded_sleep_stages = [sleep_stage for (idx, val) in enumerate(duration) for sleep_stage in sleep_stages[idx] * (val // window_length)]

    return unfolded_sleep_stages

parse_hypnogram("/home/x4kuang/sleep_staging/sleep_edf_datasets", "csv", "txt", 30)
