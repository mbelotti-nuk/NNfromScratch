import numpy as np
from .base import NeuralModule, Tensor
from opt_einsum import contract

class NeuralLayer(NeuralModule):
    def __init__(self, in_dim, out_dim):
        # layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = Tensor( np.random.randn(in_dim, out_dim) )
        self.B = Tensor( np.zeros(out_dim) )
        self.train = True
        self.zero_grads()
        self.init_weights()
        
    def init_weights(self):
        fan_in = self.W.shape[-1]
        std = np.sqrt(2/fan_in)
        self.W = Tensor( np.random.randn(*self.W.shape) * std )

    def __call__(self, x):
        return x @ self.W + self.B
    
    def backward(self, x:Tensor, h:Tensor, weight_decay=0):
        if self.train:
            self.W.grad += x.T @ h.grad + weight_decay * self.W
            self.B.grad += np.sum(h.grad, axis=0) + weight_decay * self.B
            x.grad += h.grad @ self.W.T
        return 
    
    def zero_grads(self):
        # derivatives
        self.W.grad.fill(0)
        self.B.grad.fill(0)

    def update_params(self, lr):
        self.W = self.W - lr*self.W.grad
        self.B = self.B - lr*self.B.grad
    
    def __matmul__(self, Y:np.array):
        return Y @ self.W
    

class AvgPool(NeuralModule):
    def __init__(self, k_size, stride=(1,1), padding=(0,0)):
        super().__init__()
        self.k_size_h, self.k_size_w = k_size
        self.stride = stride
        self.pad_h, self.pad_w = padding

    def __call__(self, x:Tensor) -> Tensor:

        n, c, h, w = x.shape
        out_h = (h + 2* self.pad_h - self.k_size_h) // self.stride[0] + 1
        out_w = (w + 2* self.pad_w - self.k_size_w) // self.stride[1] + 1

        windows = self.get_windows(x, (n, c, out_h, out_w), self.k_size_h, self.pad_h, self.stride[0])

        # Mean of every sub matrix, computed without considering the padd(np.nan)
        out_x = Tensor( np.nanmean(windows, axis=(4, 5)) )

        return out_x
    
    def get_windows(self,x:Tensor, output_size, kernel_size, padding=0, stride=1, dilate=0):
        working_input = x
        working_pad = padding
        # dilate the input if necessary
        if dilate != 0:
            working_input = np.insert(working_input, range(1, x.shape[2]), 0, axis=2)
            working_input = np.insert(working_input, range(1, x.shape[3]), 0, axis=3)

        # pad the input if necessary
        if working_pad != 0: 
            working_input = np.pad(working_input, pad_width=((0,), (0,), (working_pad,), (working_pad,)), mode='constant', constant_values=(0.,))

        in_b, in_c, out_h, out_w = output_size
        out_b, out_c, _, _ = x.shape
        batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides

        return np.lib.stride_tricks.as_strided(
            working_input,
            (out_b, out_c, out_h, out_w, kernel_size, kernel_size),
            (batch_str, channel_str, stride * kern_h_str, stride * kern_w_str, kern_h_str, kern_w_str),
            writeable=False
        )   
    

    def backward(self, x:Tensor, y:Tensor, weight_decay:float):
        n, c, out_h, out_w = y.shape
        _, _, in_h, in_w = x.shape
        
        windows = self.get_windows(y.grad, output_size=(n, c, in_h, in_w), kernel_size=self.k_size_h, padding=self.pad_h, dilate=0)
        grad_per_pixel = 1 / (self.k_size_h * self.k_size_w)
        x.grad += np.einsum( 'bchwkl->bchw', windows ) * grad_per_pixel

        return
    



class ConvolutionalLayer(NeuralModule):
    def __init__(self, in_ch, out_ch, k_size, stride=(1,1), padding=(0,0)):
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.k_size_h, self.k_size_w = k_size
        self.stride = stride

        n = out_ch * self.k_size_h * self.k_size_w
        self.pad_h, self.pad_w = padding

        self.train = True

        # for each output channel, there is a 3D matrix of size  in_ch x k_size x k_size 
        self.K = Tensor( np.random.randn(out_ch, in_ch, self.k_size_h, self.k_size_w) * np.sqrt(2/n) )
        self.B = Tensor( np.zeros( (1, out_ch, 1, 1)) )

        # einstein summation indices for convolutions
        self.conv_modes = {'forward':           'bihwkl,oikl->bohw', 
                           'backpropagation':   'bohwkl,oikl->bihw',  
                           'kernel_back':       'bihwkl,bohw->oikl' } 


    def __call__(self, x:Tensor):
        n, c, h, w = x.shape
        out_h = (h - self.k_size_h + 2 * self.pad_h) // self.stride[0] + 1
        out_w = (w - self.k_size_w + 2 * self.pad_w) // self.stride[1] + 1

        windows = self.get_windows(x, (n, c, out_h, out_w), self.k_size_h, self.pad_h, self.stride[0])

        out = np.einsum(self.conv_modes['forward'], windows, self.K)

        # add bias to kernels
        out += self.B

        self.cache = windows

        return Tensor(out)


    def backward(self, x:Tensor, h:Tensor, weight_decay):
        if self.train:
            windows = self.cache

            padding = self.k_size_h - 1 if self.pad_h == 0 else self.pad_h

            dout_windows = self.get_windows(h.grad, x.shape, self.k_size_h, padding=padding, stride=1, dilate=self.stride[0] - 1)
            rot_kern = np.rot90(self.K, 2, axes=(2, 3))

            self.B.grad += np.sum(h.grad, axis=(0, 2, 3)).reshape(self.B.grad.shape)
            self.K.grad += np.einsum(self.conv_modes['kernel_back'], windows, h.grad)
            x.grad += np.einsum(self.conv_modes['backpropagation'], dout_windows, rot_kern)

        return 

    def get_windows(self,x:Tensor, output_size, kernel_size, padding=0, stride=1, dilate=0):
        working_input = x
        working_pad = padding

        stride = self.stride[0] if dilate == 0 else 1

        # dilate the input if necessary
        if dilate != 0:
            working_input = np.insert(working_input, range(1, x.shape[2]), 0, axis=2)
            working_input = np.insert(working_input, range(1, x.shape[3]), 0, axis=3)

        # pad the input if necessary
        if working_pad != 0: 
            working_input = np.pad(working_input, pad_width=((0,), (0,), (working_pad,), (working_pad,)), mode='constant', constant_values=(0.,))

        in_b, in_c, out_h, out_w = output_size
        out_b, out_c, _, _ = x.shape
        batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides

        return np.lib.stride_tricks.as_strided(
            working_input,
            (out_b, out_c, out_h, out_w, kernel_size, kernel_size),
            (batch_str, channel_str, stride * kern_h_str, stride * kern_w_str, kern_h_str, kern_w_str),
            writeable=False
        )


    def zero_grads(self):
        # derivatives
        self.K.grad.fill(0)
        self.B.grad.fill(0)

    def update_params(self, lr):
        self.K = self.K - lr*self.K.grad
        self.B = self.B - lr*self.B.grad

    

class Dropout(NeuralModule):
    def __init__(self, p=0.2):
        self.p = p
        if self.p == 0:
            self.p += 1e-6
        if self.p == 1:
            self.p -= 1e-6
    
    def __call__(self, x:Tensor):
        self.mask = (np.random.rand(*x.shape) > self.p) / self.p 
        return x * self.mask
    
    def backward(self, x:Tensor, h:Tensor, weight_decay:float):
        x.grad = h.grad * self.mask
        return
