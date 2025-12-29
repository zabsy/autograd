import math
import numpy as np
import matplotlib.pyplot as plt
import random

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
    build_topo(node)
    return topo

# Directed acyclic graph to represent neural network
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
        out = Value(self.data ** other, (self,), f'**{other}')
    
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
    
        out._backward = _backward
        return out



    def __rpow__(self, other):
        return Value(other) ** self


    
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
    
    def relu(self):
        out = Value(self.data if self.data > 0 else 0, (self,), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out


    
    def tanh(self):
        n = self.data
        t = (math.exp(2*n) - 1)/(math.exp(2*n) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward(): 
            self.grad += (1 - t*t) * out.grad # Given that derivative of tanh is (1-t^2)

        out._backward = _backward

        return out
    
    def sigmoid(self):
        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    
    def backward(self):
        
        graph = _topological_graph(self)
        self.grad = 1.0
        for node in reversed(graph):
            node._backward()
        




