from dataclasses import dataclass
from typing import List, Optional

from core.math_util import *

@dataclass
class Neuron:
    weights: List[float]
    bias: float
    delta: Optional[float]
    output: Optional[float]

    def set_output(self, output: Optional[float]) -> None:
        """Sets the output value of the neuron.

        Args:
            output (Optional[float]): The result of the neuron's activation function
        """
        self.output = output

    def set_delta(self, error: float) -> None:
        """Calculates and sets the delta value for the neuron, used in backpropagation to adjust weights.  

        Args:
            error (float): The error term used for weight adjustments.
        """
        self.data = error * sigmoid_derivative(self.output)
        
    def weighted_sum(self, inputs: List[float]) -> float:
        """Computes the weighted sum of inputs and biases for the neuron.

        Args:
            inputs (List[float]): Data inputted from the input layer into the network.

        Returns:
            float: The weighted sum of inputs and bias.
        """
        ws = self.bias
        for i in range(len(self.weights)):
            ws += self.weights[i] * inputs[i]
        return ws
    
    def activate(self, inputs: List[float]) -> float:
        """Computes the neuron's activation by applying the sigmoid function to the weighted sum of inputs.

        Args:
            inputs (List[float]): Data inputted from the input layer into the network.

        Returns:
            float: The activated output of the neuron.
        """
        output = sigmoid(self.weighted_sum(inputs))
        self.set_output(output)
        return output
