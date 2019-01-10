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
                                    confidenceRange: [[low, high], ...],
                                    targets: [[['A', 2], ['B', 2]], ...],
                                  }

    Returns:
        ret_dict                : result dictionary {
                                    (c_start, c_end): <indices>,
                                }
    """

    import math

    indices = np.arange(predictions.shape[0])

    ret_dict = {}

    for confidence_range, targets in zip(config['confidenceRange'],
                                         config['targets']):
        c_start, c_end = confidence_range

        tmp_list = []

        inf_norm = np.linalg.norm(predictions_proba, ord=math.inf, axis=1)
        confidence_indices = indices[np.where((inf_norm > c_start) &
                                              (inf_norm <= c_end))]

        for tup in targets:
            # tup = [target, count]
            target_indices = indices[np.where(predictions == tup[0])]

            intersection = intersection_of_two_sorted_list(
                confidence_indices, target_indices)

            if tup[1] > len(intersection):
                print(
                    "Warning: required number of targets ({}) is larger than the number of satisfying targets in the dataset ({}), using {} instead".
                    format(tup[1], len(intersection), len(intersection)))

            tmp_list.append((tup[0], intersection[:tup[1]]))

        ret_dict[(c_start, c_end)] = tmp_list

    return ret_dict
