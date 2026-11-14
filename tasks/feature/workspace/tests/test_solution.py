import unittest
from solution import unique_in_order


class PublicTests(unittest.TestCase):
    def test_order(self):
        self.assertEqual(unique_in_order([3, 1, 3, 2]), [3, 1, 2])
