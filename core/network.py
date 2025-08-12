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
        for layer_idx in range(len(self.hidden_layers)):
            layer = self.hidden_layers[layer_idx]
            previous_layer_outputs: List[float] = (inputs if layer_idx == 0 else self.hidden_layers[layer_idx - 1].all_outputs)
            for neuron in layer.neurons:
                self.update_weights_in_a_layer(previous_layer_outputs, neuron)

    def update_weights_in_a_layer(self, previous_layer_outputs: List[float], neuron: Neuron) -> None:
        for idx in range(len(previous_layer_outputs)):
            neuron.weights[idx] -= (self.learning_rate * neuron.delta * previous_layer_outputs[idx])
            neuron.bias -= self.learning_rate * neuron.delta
    