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
        self.output = output

    def set_delta(self, error:float) -> None:
        self.data = error * sigmoid_derivative(self.output)
