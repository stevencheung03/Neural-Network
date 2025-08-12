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
    
    def back_propagate(self, inputs: List[float], errors: List[float]) -> None:
        for index, neuron in enumerate(self.output_layers.neurons):
            neuron.set_delta(errors[index])
        for layer_idx in reversed(range(len(self.hidden_layers))):
            layer = self.hidden_layers[layer_idx]
            next_layer = (self.output_layers if layer_idx == len(self.hidden_layers) - 1 else self.hidden_layers[layer_idx + 1])
            for neuron_idx, neuron in enumerate(layer.neurons):
                error_from_next_layer = next_layer.total_delta(neuron_idx)
                neuron.set_delta(error_from_next_layer)
        self.update_weights_for_all_layers(inputs)
    
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

    def train(self, num_epoch: int, num_outputs: int, training_set: List[List[float]], training_output: List[float]) -> None:
        for epoch in range(num_epoch):
            sum_error = 0.0
            for idx, row in enumerate(training_set):
                expected = [0 for _ in range(num_outputs)]
                expected[training_output[idx]] = 1
                actual = self.feed_forward(row)
                errors = self.derivative_error_to_output(actual, expected)
                self.back_propagate(row, errors)
                sum_error += self.mse(actual, training_output)
            print(f"Mean Squared Error: {sum_error}")
            print(f"Epoch = {epoch}")
    
    def predict(self, inputs: List[float]) -> int:
        outputs = self.feed_forward(inputs)
        return outputs.index(max(outputs))
    
    def mse(self, actual: List[float], expected: List[float]) -> float:
        return sum((actual[i] - expected[i]) ** 2 for i in range(len(actual))) / len(actual)
    