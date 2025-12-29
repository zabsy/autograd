import math
import numpy as np
import matplotlib.pyplot as plt
import random

# Value is a node in the graph
class Value:

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0 # Initialize by assuming no effect on final node
        self._backward = lambda: None # Base case of leaf nodes, no gradient to propogate back
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if (isinstance(other, Value)) else Value(other) # Handle case where other is not a Value
        
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward(): # Function to propogate gradient
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if (isinstance(other, Value)) else Value(other) # Handle case where other is not a Value
        out = Value(self.data * other.data, (self, other), '*')

        def _backward(): # Function to propogate gradient
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    
    def __rmul__(self, other): # other * self case
        return self * other
    
    def __truediv__(self, other): # self / other
        return self * other**-1
    
    def __neg__(self): # -self
        return self * -1
    
    def __sub__(self, other): # self - other
        return self + (-other)

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    
    def tanh(self):
        n = self.data
        t = (math.exp(2*n) - 1)/(math.exp(2*n) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward(): 
            self.grad += (1 - t*t) * out.grad # Given derivative of tanh is (1-t^2)

        out._backward = _backward

        return out
    
    def backward(self):

        # Build topological graph
        def _topological_graph(node):
            topo = []
            visited = set()
            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for i in v._prev:
                        build_topo(i)
                    topo.append(v)
            return topo
        
        graph = _topological_graph(self)
        self.grad = 1.0
        for node in reversed(graph):
            node._backward()
        

class Neuron:

    def __init__(self, nin):
        # Generate random weights and bias
        self.weights = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1,1)) 

    def __call__(self, x): # Neuron(x)
        # w * x + b
        act = sum(wi*xi for wi, xi in zip(self.weights, x)) + self.bias # dot product of w and x, + b
        out = act.tanh()
        return out
    
    def __parameters__(self):
        return self.weights + [self.bias]
    
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs
    
    def __parameters__(self):
        params = []
        for neuron in self.neurons:
            p = neuron.parameters()
            params.extend(p)
        return params


class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range (len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            p = layer.parameters()
            params.extend(p)
        return params


