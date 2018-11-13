import pandas as pd


def drop_features(df, features_to_remove=[], row_index_included=True):
    """Return a copy of the data frame with the features specified removed

    :param df:                  pandas data frame
    :param features_to_remove:  a list of feature names to remove from the data frame
    :param row_index_included:  whether the data frame has a row index column
    :return:
        new_df                  the data frame with the features removed
        indices_kept            indices of features that are kept
        indices_removed         indices of features that are removed
    """

    feature_names = df.columns
    if row_index_included:
        feature_names = feature_names[1:]

    indices = [feature_names.get_loc(val) for val in features_to_remove]

    features_to_keep = [(idx, val) for idx, val in enumerate(feature_names) if idx not in set(indices)]

    data = df[[val for idx, val in features_to_keep]].copy()

    return data, [idx for idx, val in features_to_keep], indices