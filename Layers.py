import numpy as np

class Layer():
    def __init__(self, input, neuronNumber):
        self.input = input
        self.neuronNumber = neuronNumber
        self.forward()
        
# Simple sequential model dense layer, where each node will return the sum of biases + linear combination of weights and inputs
class Dense(Layer):
    # The output can be accessed via self.output
    def forward(self):
        # The size of a single feature, where input is a list of features
        input_size = len(self.input[0])
        
        # Creates numpy array of random weights per input and intial bias of zero
        self.weights = np.random.randn(input_size, self.neuronNumber)
        self.bias = np.zeros((1, self.neuronNumber))
        
        self.output = np.dot(self.input, self.weights) + self.bias
    
    # Returns the partial derivative gradient of the input, weights, and biases. Note, the dvalues comes from the activation function
    # Furthermore, all the gradients for each individual node will be summed up for the entire layer
    # Can be accessed via self.dinput, self.dweights, and self.dbaises
    def backward(self, dvalues):
        # The partial derivative of weights with respect to inputs is the inputs (will be multiplied with dvalues, chain rule).
        # Layer summation done via the dot product
        self.dweights = np.dot(dvalues, self.weights.T)
        
        # The partial derivative of weights with respect to inputs is the inputs (will be multiplied with dvalues, chain rule).
        # Layer summation done via the dot product
        self.dweights = np.dot(self.inputs.T, dvalues)
        
        # The derivative of the baises is 1 (will be multiplied by dvalues, chain rule).
        # Layer summation done over dvalues (since derivative of bias is 1, no effect)
        self.dbias = np.sum(dvalues, axis=0, keepdims=True)