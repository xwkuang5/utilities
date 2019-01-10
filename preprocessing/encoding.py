import bisect
import numpy as np


class EncodingHelper:
    def __init__(self,
                 feature_names,
                 categorical_features,
                 label_encoder,
                 feature_label_encoder,
                 feature_one_hot_encoder=None):
        """

        Args:
            feature_names           : list of strings
            categorical_features    : list of ints
            label_encoder           : LabelEncoder object: class name => class label
            feature_label_encoder   : dictionary: feature index => LabelEncoder object: feature level => feature label
                                      One reason why a label encoder is used to encode the feature is that OneHotEncoder
                                      used to assume that the input features take on values in the range [0,max(values))
            feature_one_hot_encoder : OneHotEncoder object: label_encoded_row => one_hot_encoded_row
        """

        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.continuous_features = [
            idx for idx in range(len(self.feature_names))
            if idx not in categorical_features
        ]
        self.label_encoder = label_encoder
        self.feature_label_encoder = feature_label_encoder
        self.feature_one_hot_encoder = feature_one_hot_encoder

        acc, sep_id, sep_index = self._get_cumulative_array()

        self.acc = acc
        self.sep_index = sep_index
        self.sep_id = sep_id

    def get_feature_index_and_value(self, feature_id):
        # feature id starts from 0 so bisect_left is fine
        bisect_index = bisect.bisect_left(self.acc, feature_id)
        """
        self.acc[bisect_index] stores the endpoint (exclusive)
        if self.acc[bisect_index] == feature_id, it means that feature_id should be put
        in index bisect_index + 1. On the other hand, if the equality does not hold, it
        means that feature_id is in the range covered by bisect_index-1 and bisect_index
        """
        bisect_index = bisect_index + 1 if self.acc[
            bisect_index] == feature_id else bisect_index

        # continuous features
        if feature_id >= self.sep_id:
            feature_index = self.continuous_features[bisect_index
                                                     - self.sep_index]
            return feature_index, None
        # categorical features
        else:
            feature_index = self.categorical_features[bisect_index]

            feature_label = feature_id - self.acc[bisect_index -
                                                  1] if bisect_index > 0 else feature_id

            feature_value = self.feature_label_encoder[feature_index].classes_[
                feature_label]

            return feature_index, feature_value

    def _get_cumulative_array(self):
        """Non-categorical features are always stacked to the right of the matrix.
        """
        num_features = len(self.feature_names)
        categorical_feature_categories = self.feature_one_hot_encoder.categories_
        acc = np.empty(num_features, dtype=np.int64)

        fea_idx = 0

        for idx in self.categorical_features:

            index_in_categorical_features = self.categorical_features.index(
                idx)
            feature_count = len(
                categorical_feature_categories[index_in_categorical_features])

            if fea_idx == 0:
                acc[fea_idx] = feature_count
            else:
                acc[fea_idx] = feature_count + acc[fea_idx - 1]

            fea_idx += 1

        sep_index = fea_idx
        sep_id = acc[fea_idx - 1]

        for _ in self.continuous_features:

            acc[fea_idx] = acc[fea_idx - 1] + 1
            fea_idx += 1

        return acc, sep_id, sep_index
