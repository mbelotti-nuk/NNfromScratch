import numpy as np
from .base import NeuralModule, Tensor

class Relu(NeuralModule):
    def __init__(self):
        pass
    def __call__(self, x:Tensor):
        x[x<0] = 0
        return x
    def backward(self, x:Tensor, y:Tensor, weight_decay):
        x.grad += y.grad * (x>0).astype(float)
        return 
    

class Sigmoid(NeuralModule):
    def __init__(self):
        self.x = None
    def __call__(self, logits):
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for stability
        probabilities = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        return probabilities
    

class Flatten(NeuralModule):
    def __init__(self):
        super().__init__()
    
    def __call__(self, x:Tensor):
        self.x_shape = x.shape
        return x.reshape(self.x_shape[0],-1)
    
    def backward(self, x:Tensor, y:Tensor, weight_decay=0):
        x.grad += y.grad.reshape(self.x_shape)
        return