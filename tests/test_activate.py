import unittest

from neuron import *

class TestActivate(unittest.TestCase):
    def test_activate_case_0(self):
        weights = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
        bias = 0.5
        delta = 0
        output = 0
        neuron = Neuron(weights, bias, delta, output)
        data_input = [2.78, 2.55, 2.78, 2.55, 2.78, 2.55, 2.78, 2.55]
        result = neuron.activate(data_input)
        self.assertAlmostEqual(result, 0.9999567)

    def test_activate_case_1(self):
        weights = [0, 0]
        bias = 1.0
        delta = 0
        output = 0
        neuron = Neuron(weights, bias, delta, output)
        data_input = [10, -10, 5]
        result = neuron.activate(data_input)
        self.assertAlmostEqual(result, 0.7310585786)

    def test_activate_case_2(self):
        weights = [0.5, -0.5, 1.0]
        bias = 0
        delta = 0
        output = 0
        neuron = Neuron(weights, bias, delta, output)
        data_input = [2, 4, 6]
        result = neuron.activate(data_input)
        self.assertAlmostEqual(result, 0.9933071491)

    def test_activate_case_3(self):
        weights = [-1, -2, -3]
        bias = -5.0
        delta = 0
        output = 0
        neuron = Neuron(weights, bias, delta, output)
        data_input = [-1, -2, -3]
        result = neuron.activate(data_input)
        self.assertAlmostEqual(result, 0.9998766054)

    def test_activate_case_4(self):
        weights = [1e6, -1e6, 1e6]
        bias = 1e6
        delta = 0
        output = 0
        neuron = Neuron(weights, bias, delta, output)
        data_input = [1, 1, 1]
        result = neuron.activate(data_input)
        self.assertAlmostEqual(result, 1.0)

    def test_activate_case_5(self):
        weights = []
        bias = 1.0
        delta = 0
        output = 0
        neuron = Neuron(weights, bias, delta, output)
        data_input = []
        result = neuron.activate(data_input)
        self.assertAlmostEqual(result, 0.2689414214)

    def test_activate_case_5(self):
        weights = [0.5, 1.5]
        bias = 2.0
        delta = 0
        output = 0
        neuron = Neuron(weights, bias, delta, output)
        data_input = [1.0]
        with self.assertRaises(IndexError):
            neuron.activate(data_input)

if __name__ == "__main__":
    unittest.main()
    