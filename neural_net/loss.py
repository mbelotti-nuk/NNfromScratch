import  numpy as np
from .base import Tensor

class CrossEntropyLoss:
    def __init__(self):
        pass

    def __call__(self, logits:Tensor, target:Tensor)->Tensor:
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for stability
        probabilities = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        real = np.clip(target, 1e-9, 1.0)  # Avoid log(0)
        return  - np.sum(real * np.log(probabilities))/real.shape[0]
    
    def backward(self, logits, y):
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for stability
        probabilities = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        n = y.shape[1]
        logits.grad += (probabilities-y)/n
        return