from dataclasses import dataclass
from typing import List, Optional

from core.neuron import *

@dataclass
class Layer:
    neurons: List[Neuron]

    def all_outputs(self) -> List[float]:
        """Retrieve the output values of all neurons in the layer.

        Returns:
            List[float]: A list containing the output values of all neurons.
        """
        return [neuron.output for neuron in self.neurons]
    
    def activate_neurons(self, input: List[float]) -> List[float]:
        """Activates all neurons in the layer using the given input and returns their outputs.

        Args:
            input (List[float]): A list of input values to be fed into the neurons.

        Returns:
            List[float]: A list containing the output values of all activated neurons.
        """
        return [neuron.activate(input) for neuron in self.neurons]
    
    def total_delta(self, previous_layer_neuron: int) -> List[float]:
        """Computes the total delta contribution from this layer to a specific neuron in the previous layer. 
        This is calculated as the sum of the product of each neuron's delta and its corresponding weight connected to the specified neuron.  

        Args:
            previous_layer_neuron (int): Index of the neuron in the previous layer.

        Returns:
            List[float]: The total delta for the specified neuron in the previous layer.
        """
        return sum(neuron.weights[previous_layer_neuron] * neuron.delta for neuron in self.neurons)
