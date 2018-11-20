import unittest
import numpy as np

from evaluation.selection import select_from_datasets


class TestSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls._predictions = np.array(['A'] * 3 + ['B'] * 3 + ['C'] * 3)
        cls._predictions_proba = np.array([
            # A
            [1, 0, 0],
            [0.8, 0.1, 0.1],
            [0.4, 0.3, 0.3],
            # B
            [0, 1, 0],
            [0.2, 0.6, 0.2],
            [0.3, 0.5, 0.2],
            # C
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ])

    def testSingleTarget(self):

        self.assertEqual(
            {
                (0.6, 1): [('B', [3])]
            },
            select_from_datasets(TestSelection._predictions,
                                 TestSelection._predictions_proba, {
                                     'confidence_range': [[0.6, 1]],
                                     'targets': [('B', 3)]
                                 }))

        self.assertEqual(
            {
                (0.6, 1): [('A', [0, 1])]
            },
            select_from_datasets(TestSelection._predictions,
                                 TestSelection._predictions_proba, {
                                     'confidence_range': [[0.6, 1]],
                                     'targets': [('A', 3)]
                                 }))

        self.assertEqual(
            {
                (0.6, 1): [('C', [6, 7, 8])]
            },
            select_from_datasets(TestSelection._predictions,
                                 TestSelection._predictions_proba, {
                                     'confidence_range': [[0.6, 1]],
                                     'targets': [('C', 3)]
                                 }))

    def testTwoTargets(self):
        self.assertEqual({
            (0.6, 1): [('A', [0, 1]), ('B', [3])]
        },
                         select_from_datasets(
                             TestSelection._predictions,
                             TestSelection._predictions_proba, {
                                 'confidence_range': [[0.6, 1]],
                                 'targets': [('A', 2), ('B', 3)]
                             }))
        self.assertEqual({
            (0.8, 1): [('A', [0]), ('B', [3])]
        },
                         select_from_datasets(
                             TestSelection._predictions,
                             TestSelection._predictions_proba, {
                                 'confidence_range': [[0.8, 1]],
                                 'targets': [('A', 2), ('B', 3)]
                             }))

    def testThreeTargets(self):
        self.assertEqual({
            (0.9, 1): [('A', [0]), ('B', [3]), ('C', [6, 7, 8])]
        },
                         select_from_datasets(
                             TestSelection._predictions,
                             TestSelection._predictions_proba, {
                                 'confidence_range': [[0.9, 1]],
                                 'targets': [('A', 2), ('B', 3), ('C', 3)]
                             }))


if __name__ == '__main__':
    unittest.main()
