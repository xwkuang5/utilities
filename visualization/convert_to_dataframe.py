import numpy as np
import pandas as pd


def convert_2d_array_to_df(array, row_name, row_labels, column_name,
                           indices_name, indices_labels):
    """Convert a 2D numpy array into a pandas dataframe

    Assume that the columns index the measures.

    Arguments:
        array               : (m, n) array of int/float,
        row_name            : str, column name of the first axis in the data frame
        row_labels          : [str], list of row labels to use for the row column
        column_name         : str, column name of the second axis in the data frame
        indices_name        : str, column name of the indices
        indices_labels      : [str/int], list of indices labels to use for the indices

    Return:
        df                  : data frame
    """

    assert array.shape[0] == len(
        row_labels
    ), "Number of rows is not the same as the number of row labels provided"

    m, n = array.shape

    flat_array = array.flatten()

    rows_column = np.array(
        [[row_labels[idx]] * n for idx in range(m)]).flatten()

    indices_column = np.array([indices_labels for _ in range(m)]).flatten()

    return pd.DataFrame({
        row_name: pd.Series(rows_column),
        column_name: pd.Series(flat_array),
        indices_name: pd.Series(indices_column)
    })
