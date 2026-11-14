import unittest
from solution import average


class Acceptance(unittest.TestCase):
    def test_edges(self):
        self.assertIsNone(average([]))
        self.assertEqual(average([0]), 0)
        self.assertAlmostEqual(average([-2.5, 1.5]), -0.5)
        values = [1, 2, 3]
        self.assertEqual(average(values), 2)
        self.assertEqual(values, [1, 2, 3])
