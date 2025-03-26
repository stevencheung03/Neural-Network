from dataclasses import dataclass
from typing import List, Optional

from neuron import *

@dataclass
class Layer:
    neurons: List[Neuron]

    def all_outputs(self, inputs: List[float]) -> List[float]:
        return [neuron.output for neuron in self.neurons]
    
    def activate_neurons(self, input: List[float]) -> List[float]:
        return [neuron.activate(input) for neuron in self.neurons]
    
    def total_delta(self, previous_layer_neuron: int) -> List[float]:
        return sum(neuron.weights[previous_layer_neuron] * neuron.delta for neuron in self.neurons)
