import unittest
from solution import average


class PublicTests(unittest.TestCase):
    def test_regular(self):
        self.assertEqual(average([2, 4]), 3)

    def test_empty(self):
        self.assertIsNone(average([]))
