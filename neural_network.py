import numpy as np
import data_prep as dp
import shiv_ai as nn

class Layer():
    pass

class Activation():
    def __init__(self, input):
        self.forward(input)
        
class Loss:
    def __init__(self, output, target, oneshot=False):
        self.calculate(output, target, oneshot)
    def calculate(self, output, target, oneshot=False):
        batchLoss = self.forward(output, target, oneshot)
        dataLost = np.mean(batchLoss)
        return dataLost

# Implementation of simple hidden dense layer
# The constuctor will automatically calculate the output values per neuron (found in self.output)
class Dense(Layer):
    def __init__(self, input, neuronNumber):
        input_size = len(input[0])
        self.input = input
        # Creates numpy array of random weights per input and intial bias of zero
        self.weights = np.random.randn(input_size, neuronNumber)
        self.bias = np.zeros((1, neuronNumber))
        self.forward(self.input)
    # The summation of the weights and inputs + bias (found in self.output)
    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.bias
    # This will be the backpropogation function, returning the derivative of the outputs of the -
    # - forward function, which would result in the derivative weights (dweights) being multiplied - 
    # - the back backpropogation of the activation (dvalues) and the derivative biases
    # To simplify batch testing, we will add the dvalues of a layer for each individual feature
    def backward(self, dvalues):
        # Will not go too deep, but derivative of biases = 1, since only 1 bias/neuron and biases are cosntant -
        # - thus we will just sum the dvalues (axis=0  since we want to add rows, as they are the individual features)
        self.dbaises = np.sum(dvalues, axis=0, keepdims=True)
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dinputs = np.dot(dvalues, self.weights.T)

# Implementation of the Rectified Linear Unit actication function -
# -which is simpler and quicker than the sigmoid function (found in self.output)
# Putting in the input in the constuctor will automatically perform the activation
class ReLU(Activation):
    # Pretty simple implementation
    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, input)
    # The ReLU function derivative is essenstially, if the value in matrix > 0, the dvalue = 1, if value in matrix <= 0, the dvalue = 0
    def backward(self, dvalues):
        self.dReLU = dvalues.copy()
        self.dReLU[self.input <= 0] = 0
        

# Implementation of the softmax activation function, which will output the -
# - probabilites of each output neuron (found in self.output)
# Putting in the input in the constuctor will automatically perform the activation
class SoftMax(Activation):
    def forward(self, input):
        self.output = self.normalize(input)
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
    def normalize(self, input):
        max = np.max(input, axis=1, keepdims=True)
        X = np.exp(input - max)
        rowSum = np.sum(input, axis=1, keepdims=True)
        return X / rowSum

# The implementation of categorical cross entropy
class CrossEntropy(Loss):
    def forward(self, softMaxOut, target, oneshot=False):
        size = len(softMaxOut)
        softMaxOutClip = np.clip(softMaxOut, 1e-7, 1 - 1e-7)
        if(oneshot):
            targetCorreletion = np.sum(softMaxOutClip * target, axis=1)
        else:
            targetCorreletion = softMaxOutClip[range(size), target]
            
        crossEntropy = -np.log(targetCorreletion)
        return crossEntropy
    
    def backward(self, dvalues, y, oneshot=False):
        n = len(dvalues)
        y_len = len(dvalues[0])
        if(not oneshot):
            y = dp.convertLabel2OneShot(y)
            
        self.dinputs = (-y / dvalues) / n
    
    def accuracy(softMax, target, oneshot=False):
        prediction = np.argmax(softMax, axis=1)
        accurate = np.mean(prediction == target)
        return accurate
    
#Joined the softmax and cross entropy for faster backpropogation
class SoftMaxNCrossEntropy():
    def __init__(self):
        self.activation = SoftMax() 
        self.loss = CrossEntropy()
    def forward(self, inputs, y):
        self.output = self.activation.output
        return self.loss.calculate(self.output, y)
    def backward(self, dvalues, y, oneshot=False):
        n = len(dvalues)
        if(oneshot):
            y = np.argmax(y, axis=1)
            
        self.dinputs = dvalues.copy()
        self.dinputs[range(n), y] -= 1
        self.dinputs /= n