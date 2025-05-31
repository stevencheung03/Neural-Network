import unittest

from core.math_util import *
from core.neuron import Neuron
from core.layer import Layer
from core.network import Network

class TestFeedForward(unittest.TestCase):
    def test_feed_forward(self):
        # Define neurons with simple weights and biases

        # Input vector
        inputs = [2.78, 2.55]

        # Hidden Layer: 2 neurons
        hidden1 = Neuron(weights=[0.1, 0.2], bias=0.5, delta=None, output=None)
        hidden2 = Neuron(weights=[0.3, 0.4], bias=0.12, delta=None, output=None)
        hidden_layers = Layer(neurons=[hidden1, hidden2])

        # Output Layer: 2 neurons
        output1 = Neuron(weights=[0.5, 0,6], bias=0.5, delta=None, output=None)
        output2 = Neuron(weights=[0.7, 0.8], bias=0.5, delta=None, output=None)
        output_layer = Layer(neurons=[output1, output2])

        # Create network
        nn = Network([hidden_layers], output_layer)

        # Run feed forward
        outputs = nn.feed_forward(inputs)

        hidden_output = sigmoid(1.0)
        expected_final = sigmoid(hidden_output * 2)

        self.assertAlmostEqual(outputs[0], expected_final, places=4)

if __name__ == '__main__':
    unittest.main()
