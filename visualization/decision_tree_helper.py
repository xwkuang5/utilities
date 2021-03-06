import numpy as np


class TreeNode:
    def __init__(self, feature_id):
        self.feature_id = feature_id
        self.threshold = None
        self.is_leaf = None
        self.class_name = None
        self.right = None
        self.left = None

    def __repr__(self):
        return "TreeNode({}), is_leaf: {}".format(self.feature_id,
                                                  self.is_leaf)

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def retrieve_path_as_predicate(root):
        """Return a list of list of predicates

        Each list of predicate represents a path from the root to the leaf

        Predicate format: (is_leaf, class_label, feature_id, threshold, relation)
        """

        if root.is_leaf:
            return [[(True, root.class_name, None, None, None)]]
        else:
            left_predicate = (False, None, root.feature_id, root.threshold,
                              '<=')
            right_predicate = (False, None, root.feature_id, root.threshold,
                               '>')

            left_branches = TreeNode.retrieve_path_as_predicate(root.left)
            right_branches = TreeNode.retrieve_path_as_predicate(root.right)

            return [[left_predicate] + branch for branch in left_branches
                    ] + [[right_predicate] + branch
                         for branch in right_branches]

    @staticmethod
    def get_predicate(feature_id, threshold, relation, encoder_helper):
        """Return a predicate represented by a tree node in the decision tree

        Args:
            feature_id          : int (feature id of the tree node, note not id of the node)
            threshold           : float, threshold used by the node to make a branching decision
            relation            : str, '<=' or '>', indicating left or right branch
            encoder_helper      : EncoderHelper
                                  from utilities.preprocessing.encoding import EncodingHelper
        """

        def predicate(row):
            feature_index, feature_value = encoder_helper.get_feature_index_and_value(
                feature_id)
            if feature_index in encoder_helper.continuous_features:
                if relation == '<=':
                    return float(row[feature_index]) <= threshold
                else:
                    return float(row[feature_index]) > threshold
            else:
                assert threshold == 0.5, "Threshold for one hot encoded data is {}, not 0.5".format(
                    threshold)
                assert feature_value is not None, "Feature value is None"

                if relation == '<=':
                    return row[feature_index] != feature_value
                else:
                    return row[feature_index] == feature_value

        return predicate


class DecisionTreeHelper:
    def __init__(self, decision_decision_tree_instance):

        self.decision_tree_instance = decision_decision_tree_instance
        self.n_nodes = self.decision_tree_instance.tree_.node_count
        self.value = self.decision_tree_instance.tree_.value
        self.children_left = self.decision_tree_instance.tree_.children_left
        self.children_right = self.decision_tree_instance.tree_.children_right
        self.feature = self.decision_tree_instance.tree_.feature
        self.threshold = self.decision_tree_instance.tree_.threshold

    def convert_array_repr_to_tree_repr(self):

        if self.n_nodes == 0:
            return None

        tree_node_dict = {}

        def get_node_from_dict(nid, feature_id):
            if nid not in tree_node_dict:
                tree_node_dict[nid] = TreeNode(feature_id)
            return tree_node_dict[nid]

        for node_id in range(self.n_nodes):
            node = get_node_from_dict(node_id, self.feature[node_id])

            if self.children_left[node_id] == self.children_right[node_id]:

                node.is_leaf = True
                node.class_name = np.argmax(self.value[node_id])

            else:

                node.is_leaf = False
                node.threshold = self.threshold[node_id]
                left_node = get_node_from_dict(
                    self.children_left[node_id],
                    self.feature[self.children_left[node_id]])
                right_node = get_node_from_dict(
                    self.children_right[node_id],
                    self.feature[self.children_right[node_id]])
                node.left = left_node
                node.right = right_node

        return tree_node_dict[0]
