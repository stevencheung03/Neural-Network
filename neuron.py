from dataclasses import dataclass
from typing import List, Optional

from math_util import *

@dataclass
class Neuron:
    weights:List[float]
    bias:float
    delta:Optional[float]
    output:Optional[float]

    def set_output(self, output:Optional[float]) -> None:
        """_summary_

        Args:
            output (Optional[float]): The result of the neuron's activation function
        """
        self.output = output

    def set_delta(self, error:float) -> None:
        """_summary_

        Args:
            error (float): Used during backpropagation process to adjust weights
        """
        self.data = error * sigmoid_derivative(self.output)
        
    def weighted_sum(self, inputs:List[float]) -> float:
        """_summary_

        Args:
            inputs (List[float]): data inputted from the input layer into the network

        Returns:
            float: weighted sum
        """
        ws = self.bias
        for i in range(len(self.weights)):
            ws += self.weighted_sum[i] * inputs[i]
        return ws
    
    def activate(self, inputs:List[float]) -> float:
        """_summary_

        Args:
            inputs (List[float]): data inputted from the input layer into the network

        Returns:
            float: activation of the inputs
        """
        output = sigmoid(self.weighted_sum(inputs))
        self.set_output
        return output
