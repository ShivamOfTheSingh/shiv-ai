import numpy as np
import data_prep as dp
import Activation as act

class Loss:
    def __init__(self, output, target, oneshot=False):
        self.calculate(output, target, oneshot)
    
    # Will calculate the loss for each value of the output of the neural network and return the average value of the summation
    def calculate(self, output, target, oneshot=False):
        batchLoss = self.forward(output, target, oneshot)
        dataLost = np.mean(batchLoss)
        return dataLost
    
# The implementation of categorical cross entropy loss function
class CrossEntropy(Loss):
    # The forward pass will calculate the loss. Since we are using oneshot, we wouldn't need to sum all the values, we only need to -
    # - use the formula for the since softmax out index which correlates with the target value.
    def forward(self, softMaxOut, target, oneshot=False):
        size = len(softMaxOut)
        softMaxOutClip = np.clip(softMaxOut, 1e-7, 1 - 1e-7)
        if(oneshot):
            targetCorreletion = np.sum(softMaxOutClip * target, axis=1)
        else:
            targetCorreletion = softMaxOutClip[range(size), target]
            
        crossEntropy = -np.log(targetCorreletion)
        return crossEntropy
    
    # Returns the derivative gradient
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
    
# Since the softmax layer and the categorical cross entropy funcction are often used simultaneously - 
# - the 2 classes will be combined to simplify backpropogation
class SoftMaxAndCrossEntropy():
    def __init__(self):
        self.softMax = act.SoftMax()
        self.loss = CrossEntropy()
    
    # Essentially does the forward pass of the softmax then the forward pass of loss function
    def forward(self, input, y, oneshot=False):
        self.softMax.forward(input)
        self.output = self.softMax.output
        return self.loss.forward(self.output, y, oneshot)
    
    # This is the difference, the partial derivative of loss and partial dereivative of softmax, with respect to the layers inputs,
    # dLoss/dLossInput and dSoftmax/dSoftmax inputs can be reduced to dLoss/dSoftmaxInputs, and an optomization can be made
    def backward(self, dvalue, y, oneshot=False):
        n = len(dvalue)
        if(oneshot):
            y = dp.convertOneShot2Label(y)
        
        self.dinputs = dvalue.copy()
        self.dinputs[range(n), y] -= 1 
        self.dinputs /= n