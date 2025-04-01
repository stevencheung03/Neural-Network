import unittest

from neuron import *

class TestWeightedSum(unittest.TestCase):
    def test_weighted_sum_case_0(self):
        weights = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
        bias = 0.5
        delta = 0
        output = 0
        neuron = Neuron(weights, bias, delta, output)
        data_input = [2.78, 2.55, 2.78, 2.55, 2.78, 2.55, 2.78, 2.55]
        result = neuron.weighted_sum(data_input)
        self.assertAlmostEqual(result, 10.047999999999998)

    def test_weighted_sum_case_1(self):
        weights = [0, 0]
        bias = 1.0
        delta = 0
        output = 0
        neuron = Neuron(weights, bias, delta, output)
        data_input = [10, -10, 5]
        result = neuron.weighted_sum(data_input)
        self.assertAlmostEqual(result, 1.0)

    def test_weighted_sum_case_2(self):
        weights = [0.5, -0.5, 1.0]
        bias = 0
        delta = 0
        output = 0
        neuron = Neuron(weights, bias, delta, output)
        data_input = [2, 4, 6]
        result = neuron.weighted_sum(data_input)
        self.assertAlmostEqual(result, 5.0)

    def test_weighted_sum_case_3(self):
        weights = [-1, -2, -3]
        bias = -5.0
        delta = 0
        output = 0
        neuron = Neuron(weights, bias, delta, output)
        data_input = [-1, -2, -3]
        result = neuron.weighted_sum(data_input)
        self.assertAlmostEqual(result, 9.0)

    def test_weighted_sum_case_4(self):
        weights = [1e6, -1e6, 1e6]
        bias = 1e6
        delta = 0
        output = 0
        neuron = Neuron(weights, bias, delta, output)
        data_input = [1, 1, 1]
        result = neuron.weighted_sum(data_input)
        self.assertAlmostEqual(result, 2e6)

    def test_weighted_sum_case_5(self):
        weights = []
        bias = 1.0
        delta = 0
        output = 0
        neuron = Neuron(weights, bias, delta, output)
        data_input = []
        result = neuron.weighted_sum(data_input)
        self.assertAlmostEqual(result, bias)

    def test_weighted_sum_case_5(self):
        weights = [0.5, 1.5]
        bias = 2.0
        delta = 0
        output = 0
        neuron = Neuron(weights, bias, delta, output)
        data_input = [1.0]
        with self.assertRaises(IndexError):
            neuron.weighted_sum(data_input)

if __name__ == "__main__":
    unittest.main()
    