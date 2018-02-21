import os
import argparse
import numpy as np


def write_arff_header_with_file_handle(file_handle, relation_name, target,
                                       data):
    """Write ARFF relation and attribute into file pointed to by file_handle

    Assumption:
        Feature value has type "numeric"

    Arguments:
        file_handle     -- file_handle pointing to the output file
        relation_name   -- name of the arff relation
        target          -- one-dimensional numpy array (m x 1) of taget
        data            -- two-diemsnional numpy array (m x n) of data
    """

    assert target.shape[0] != 0, "target data must have size > 0"
    assert target.shape[0] == data.shape[
        0], "shape of target does not match shape of data"

    block_str = ""
    relation_str = "@relation " + relation_name + "\n\n"
    block_str += relation_str

    attribute_template = "@attribute attr{} numeric\n"

    for idx in range(1, data.shape[1] + 1):
        block_str += attribute_template.format(idx)

    possible_targets = np.unique(target)
    target_str = "@attribute target {" + ",".join(
        [str(target) for target in possible_targets]) + "}\n\n"
    block_str += target_str

    file_handle.write(block_str)


def write_arff_data_with_file_handle(file_handle, target, data):
    """Write data into file pointed to by file_handle in ARFF format

    Arguments:
    file_handle     -- file_handle pointing to the output file
    target          -- one-dimensional numpy array (m x 1) of taget
    data            -- two-diemsnional numpy array (m x n) of data
    """

    assert target.shape[0] == data.shape[
        0], "shape of target does not match shape of data"

    block_str = ""
    block_str += "@data\n"
    for i in range(data.shape[0]):
        line_str = ""
        for j in range(data.shape[1]):
            line_str += str(data[i, j]) + ","
        line_str += str(target[i]) + "\n"
        block_str += line_str

    file_handle.write(block_str)


def write_ucr_data_with_filename(filename, target, data):
    """Write data into file pointed to by file_handle in UCR format

    Arguments:
    filename        -- filename of the output file
    target          -- one-dimensional numpy array (m x 1) of taget
    data            -- two-diemsnional numpy array (m x n) of data
    """

    assert target.shape[0] == data.shape[0]

    dataframe = np.hstack([target.reshape((-1, 1)), data])

    fmt_str = "%d," + ",".join(["%f"] * data.shape[1])

    np.savetxt(filename, dataframe, fmt_str)


def from_ucr_to_arff(filename, output):
    """Convert data from ucr format to arff format

    If target is binary, assume the following:
        target value of 1 will be 1
        target value of not 1 will be -1

    UCR data format:
        target,val,val,...,val

    ARFF dataformat:
        @relation_name
        @attribute
        @data

    Arguments:
    filename    -- input file name /path/to/input
    output      -- output file name /path/to/output, can be None
    """

    if os.path.exists(filename):
        if filename[-4:] != ".ucr":
            print("filename [{}] is not in ucr format, skip".format(filename))
        try:
            data = np.loadtxt(filename, dtype=np.float64, delimiter=",")
            target = np.array(data[:, 0], dtype=np.int8)
            data = data[:, 1:]

            unique_targets = np.unique(target)

            if unique_targets.shape[0] == 2:
                # remap 0 or -1 to -1
                target[target != 1] = -1

            out_filename = output if output != None else filename[:-4] + ".arff"

            with open(out_filename, "w") as f:
                relation_name = out_filename.split("/").pop()
                write_arff_header_with_file_handle(f, relation_name, target,
                                                   data)
                write_arff_data_with_file_handle(f, target, data)
            f.close()

            print("[complete] convert {} to {}".format(filename, out_filename))
        except:
            raise
            # print("Unexpected error:", sys.exc_info()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="/path/to/input")
    ap.add_argument(
        "--output",
        required=False,
        help=
        "/path/to/output, if not provided, .arff is appended to input \
        file when creating the output"
    )
    ap.add_argument("--input_format", required=True, help="ucr")
    ap.add_argument("--output_format", required=True, help="ucr")

    args = vars(ap.parse_args())

    if args["input_format"] == "ucr" and args["output_format"] == "arff":
        from_ucr_to_arff(args["input"], args["output"])


if __name__ == "__main__":
    main()
