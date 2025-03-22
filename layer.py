from dataclasses import dataclass
from typing import List, Optional

from neuron import *

@dataclass
class Layer:
    neurons: List[Neuron]
