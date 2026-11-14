import unittest
from solution import is_prime


class Acceptance(unittest.TestCase):
    def test_range(self):
        for n in range(-5, 150):
            expected = n >= 2 and all(n % d for d in range(2, n))
            self.assertEqual(is_prime(n), expected, n)
