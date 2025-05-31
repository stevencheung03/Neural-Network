from dataclasses import dataclass
from typing import List, Optional

from layer import *

@dataclass
class Network:
    hidden_layers: List[Layer]
    output_layers: Layer
    learning_rate: float

    def layers(self) -> List[Layer]:
        return self.hidden_layers + [self.output_layers]
    
    def feed_forward(self, inputs: List[float]) -> List[float]:
        for layer in self.hidden_layers:
            inputs = layer.activate_neurons(inputs)
        return self.output_layers.activate_neurons(inputs)
