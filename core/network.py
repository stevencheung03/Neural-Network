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
