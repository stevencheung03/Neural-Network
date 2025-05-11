from dataclasses import dataclass
from typing import List, Optional

from layer import *

@dataclass
class Network:
    hidden_layers: List[Layer]
    output_layers: Layer
    learning_rate: float

    @property
    def layers(self) -> List[Layer]:
        """
        Returns a list of all layers, including both hidden and output layers.

        This combines the hidden layers with the output layer into a single list,
        preserving their order of execution.

        Returns:
            List[Layer]: Ordered list containing all layers of the model.
        """
        return self.hidden_layers + [self.output_layers]
