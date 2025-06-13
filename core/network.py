from dataclasses import dataclass
from typing import List, Optional

from core.layer import *

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
    
    def derivative_error_to_output(self, actual: List[float], expected: List[float]) -> List[float]:
        return [actual[i]-expected[i] for i in range(len(actual))]
    
    def update_weights_for_all_layers(self, inputs: List[float]):
        pass

    def update_weights_in_a_layer(self, previous_layer_outputs: List[float], neuron: Neuron) -> None:
        pass
    