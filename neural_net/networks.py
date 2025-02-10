import numpy as np
from .base import NeuralModule, Tensor, NeuralModulesList

class ANN:
    def __init__(self, loss, weight_decay=1E-3):
        self.loss = loss
        self.layers = NeuralModulesList()
        self.train = True
        self.weight_decay = weight_decay
    
    def __call__(self, x:np.array):
        if self.train:
            self.intermediates = [Tensor(np.copy(x))]
        for layer in self.layers:
            x = layer(x)
            if self.train: self.intermediates.append(Tensor(np.copy(x)))
        return x
    
    def get_loss(self, y_hat:Tensor, y:Tensor):
        loss_value = self.loss(y_hat, y) 
        if self.weight_decay > 0:
            penalty = sum( [np.sum(w**2) for w in self.get_params()] )
            loss_value += (self.weight_decay/2)*penalty
        if self.train: self.y = y
        return loss_value
    
    def backward(self):
        self.y.grad = np.ones_like(self.y.grad)
        self.loss.backward(self.intermediates[-1], self.y)
        for i in reversed(range(len(self.layers))):
            self.layers[i].backward(self.intermediates[i], self.intermediates[i+1], self.weight_decay)

    def step(self, lr):
        for layer in self.layers:
            layer.update_params(lr)

    def zero_grads(self):
        for layer in self.layers:
            layer.zero_grads()

    def get_params(self):
        params = []
        for layer in self.layers:
            p = layer.get_params()
            if p!= None:
                params.extend(layer.get_params())   
        return params     
    


