import unittest
from solution import unique_in_order


class Acceptance(unittest.TestCase):
    def test_unhashable(self):
        items = [[1], [2], [1]]
        self.assertEqual(unique_in_order(items), [[1], [2]])
        self.assertEqual(items, [[1], [2], [1]])

    def test_empty(self):
        self.assertEqual(unique_in_order([]), [])
