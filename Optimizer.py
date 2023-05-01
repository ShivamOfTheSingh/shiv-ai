import numpy as np

class Optimizer():
    def __init__(self, step=0.1, decay=0, momentum=0):
        self.learningRate = step
        self.decay = decay
        self.p = momentum
        self.i = 0
       
# Implementation of the Stochastic Gradient Descent Optomizer
class SGD(Optimizer):
    def updateLayer(self, layer):
        # If momentum, we will use this algorithm to figure out the change in weight and baises
        if self.p:
            # For intial iteration of a layer, since the gradient descent hasn't "moved" the momentum will be zero
            if not hasattr(layer, 'pweights'):
                layer.pweights = np.zeros_like(layer.weights)
                layer.pbias = np.zeros_like(layer.bais)
            
            # Calculates the velocity (how much the weights will actually step by), using the current learning rate and the momentum.
            # Momentum is a hyperparameter which will save all the previous rates at a porportion (the actual value of momentum) which -
            # - is saved in layer.pweights and layer.pbias, as a runnign average. Running average will decrease as learning rate decays
            vweights = self.p * layer.pweights - self.learningRate * layer.dweights
            # This is what updated the momentums for weights, which will be used in next iteration, creating a runnign total
            layer.pweights = vweights
            
            # Same concept for baises
            vbias = self.p * layer.pbias - self.learningRate * layer.dbias
            layer.pbias = vbias
        
        # Vanilla SGD implementation
        else:
            vweights = -self.learningRate + layer.dweights
            vbias = -self.learningRate + layer.dbias
            
        layer.weights += vweights
        layer.bias += vbias
        self.i += 1
        
    def updateRate(self):
        self.currentRate = self.learningRate / (1 + self.decay * self.i)