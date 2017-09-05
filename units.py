import numpy as np


class Unit():
    
    def __init__(self, value, grad = 0.0):
        
        self.value = value
        self.grad = grad

        
class Output():
    
    def __init__(self, num_params):        
        self.weights = np.random.randn(num_params)
        self.bias = np.zeros(1)
        
    def forward(self, inputs):
        
        self.inputs = inputs
        input_vals = np.array([inp.value for inp in self.inputs])
        
        self.out = Unit(np.dot(self.weights, input_vals) + self.bias)
        
        return self.out
    
    def backward(self):
        
        for inp,weight in zip(self.inputs, self.weights):
            inp.grad += weight * self.out.grad
            
    def update(self, alpha = 0.1):
        #update bias
        self.bias += alpha * self.out.grad

        #update weights
        for i, inp in list(enumerate(self.inputs)):
            self.weights[i] += alpha * inp.value * self.out.grad
            
    def reset_grad(self):
        
        self.out.gradient = 0.0
        

class Sigmoid():
    
    def __init__(self, num_params):
        
        self.weights = np.random.randn(num_params)
        self.bias = np.zeros(1)
        
    def sigmoid(self, x):
        
        return 1.0 / (1.0 + np.exp(-x))
    
    def forward(self, inputs):
        
        self.inputs = inputs
        input_vals = np.array([inp.value for inp in self.inputs])
        
        #better to store this as needed later when updating weights
        self.inp_x_W = np.dot(self.weights, input_vals) + self.bias
        
        self.out = Unit(self.sigmoid(self.inp_x_W))
        
        return self.out
    
    def backward(self):
        
        g = self.sigmoid(self.inp_x_W)
        d_sig = g * (1 - g)
        for inp,weight in zip(self.inputs, self.weights):
            inp.grad += weight * d_sig * self.out.grad
            
    def update(self, alpha = 0.1):
        g = self.sigmoid(self.inp_x_W)
        d_sig = g * (1 - g) #gradient of sigmoid
        
        #update bias
        self.bias += alpha * d_sig * self.out.grad

        #update weights
        for i, inp in list(enumerate(self.inputs)):
            self.weights[i] += alpha * inp.value * d_sig * self.out.grad
        
    def reset_grad(self):
        
        self.out.gradient = 0.0

        
class Relu():
    
    def __init__(self, num_params):
        
        self.weights = np.random.ranf(num_params)
        self.bias = np.random.randn()

    def forward(self, inputs):

        self.inputs = inputs
        input_vals = np.array([inp.value for inp in self.inputs])
        self.out = Unit(max(0, np.dot(input_vals, self.weights) + self.bias))
        return self.out
    
    def backward(self):
        #propogate gradient to inputs
        
        for inp,weight in zip(self.inputs, self.weights):
            inp.grad += int(0 != self.out.value) * weight * self.out.grad

    def update(self, alpha = 0.01):
        
        #update bias
        self.bias += alpha * int(0 != self.out.value) * self.out.grad

        #update weights
        for i, inp in list(enumerate(self.inputs)):
            self.weights[i] += alpha * int(0 != self.out.value) * inp.value * self.out.grad
        
    def reset_grad(self):
        self.out.gradient = 0.0
