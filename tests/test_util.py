import unittest

import numpy as np

from torch_fairness.util import set_random_state


class TestSetRandomState(unittest.TestCase):
    def test_incorrect_input(self):
        self.assertRaises(ValueError, set_random_state, random_state="apple")

    def test_integer_input(self):
        out = set_random_state(random_state=1)
        self.assertTrue(isinstance(out, np.random.RandomState))

    def test_state_input(self):
        out = set_random_state(random_state=np.random.RandomState(1))
        self.assertTrue(isinstance(out, np.random.RandomState))
