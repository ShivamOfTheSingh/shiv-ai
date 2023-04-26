import numpy as np

class Loss:
    def __init__(self, output, target, oneshot=False):
        self.calculate(output, target, oneshot)
        
    def calculate(self, output, target, oneshot=False):
        batchLoss = self.forward(output, target, oneshot)
        dataLost = np.mean(batchLoss)
        return dataLost
    
# The implementation of categorical cross entropy loss function
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