import numpy as np

class Activation():
    def __init__(self, input):
        self.input = input
        self.forward()
        
# Simple activation function, the Rectified Linear Unit activation function.
class ReLU(Activation):
    # Implementation of the ReLU activation function
    # Output can be found in self.output
    def forward(self):
        self.output = np.maximum(0, self.input)
        
    # The ReLU function will output either positive integers or 0, thus the derivative of the ReLU matrix -
    # - consists of either 1's (positive integers) or zeros.
    # Gradient can be accessed via self.drelu
    def backward(self, dvalues):
        self.drelu = dvalues.copy()
        self.drelu[self.input <= 0] = 0
        
# The softmax activation function for the last output layer
class SoftMax(Activation):
    # Normalizes and returns the softmax array
    def forward(self):
        max = np.max(self.input, axis=1, keepdims=True)
        X = np.exp(self.input - max)
        rowSum = np.sum(self.input, axis=1, keepdims=True)
        self.output = X / rowSum
        
    def backward(self, dvalues):
        