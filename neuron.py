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
