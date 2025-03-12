from math import exp

def sigmoid(x:float) -> float:
    return 1/(1+exp(-x))

def sigmoid_derivative(x:float) -> float:
    return x * exp(1.0 - x)
