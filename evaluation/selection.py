import numpy as np


def intersection_of_two_sorted_list(arr1, arr2):

    len_1, len_2 = len(arr1), len(arr2)

    if len_1 == 0 or len_2 == 0:
        return []
    else:
        i, j = 0, 0
        ret = []

        while i < len_1 and j < len_2:
            if arr1[i] == arr2[j]:
                ret.append(arr1[i])
                i += 1
                j += 1
            elif arr1[i] < arr2[j]:
                i += 1
            else:
                j += 1

        return ret


def select_from_datasets(predictions, predictions_proba, config):
    """Stratefied selection of examples from the dataset

    Arguments:
        predictions             : (m, ) numpy array
        predictions_proba       : (m, c) numpy array
        config                  : configuration dictionary {
                                    confidence_range: [[low, high], ...],
                                    targets: [[('A', 2), ('B', 2)], ...],
                                    shuffled: true/false
                                  }
    """

    import math

    indices = np.arange(predictions.shape[0])

    ret_dict = {}

    for c_start, c_end in config['confidence_range']:

        tmp_list = []

        inf_norm = np.linalg.norm(predictions_proba, ord=math.inf, axis=1)
        confidence_indices = indices[np.where((inf_norm > c_start) &
                                              (inf_norm <= c_end))]

        for target, cnt in config['targets']:

            target_indices = indices[np.where(predictions == target)]

            intersection = intersection_of_two_sorted_list(
                confidence_indices, target_indices)

            if cnt > len(intersection):
                print(
                    "Warning: required number of targets ({}) is larger than the number of satisfying targets in the dataset ({}), using {} instead".
                    format(cnt, len(intersection), len(intersection)))

            size = min(cnt, len(intersection))
            tmp_list.append((target, intersection))

        ret_dict[(c_start, c_end)] = tmp_list

    return ret_dict
