import numpy as np
from typing import Iterator

class Tensor(np.ndarray):
    def __new__(cls, input_array, info=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array, dtype=np.float32).view(cls)
        # add the new attribute to the created instance
        obj.grad = np.zeros_like(input_array)
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.grad = getattr(obj, 'grad', None)

class NeuralModule:
    def __init__(self):
        pass
    def backward(self, x:Tensor, y:Tensor, weight_decay=0):
        pass
    def update_params(self, lr:float):
        pass
    def zero_grads(self):
        pass
    def get_params(self):
        return None

class NeuralModulesList:
    modules : dict[int, NeuralModule]
    def __init__(self,):
        self.modules = dict()
        self.n = 0
        pass

    def __iter__(self) -> Iterator[NeuralModule]:
        return iter(self.modules.values())

    def __next__(self) -> NeuralModule: 
        return iter(self.modules.values())

    def __getitem__(self, index) -> NeuralModule:
        return self.modules[index]

    def append(self, module:NeuralModule):
        self.modules[self.n] = module
        self.n += 1

    def extend(self, lst_modules:list[NeuralModule]):
        for module in lst_modules:
            self.modules[self.n] = module
            self.n += 1        

    def __len__(self):
        return self.n