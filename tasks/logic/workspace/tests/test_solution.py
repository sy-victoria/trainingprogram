import unittest
from solution import is_prime


class PublicTests(unittest.TestCase):
    def test_prime(self):
        self.assertTrue(is_prime(7))

    def test_square(self):
        self.assertFalse(is_prime(9))
