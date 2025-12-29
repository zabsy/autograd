from autograd import Value
import random


class Neuron:

    def __init__(self, nin, nonlin=True):
        self.nonlin = nonlin # Flag to indicate if apply activation function
        # Generate random weights and bias on initialization
        self.weights = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1,1)) 

    def __call__(self, x):
        # w * x + b
        act = self.bias
        for wi, xi in zip(self.weights, x):
            act = act + wi * xi

        out = act.relu() if self.nonlin else act
        return out
    
    def parameters(self):
        return self.weights + [self.bias]
    
class Layer:
    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs
    
    def parameters(self):
        params = []
        for neuron in self.neurons:
            p = neuron.parameters()
            params.extend(p)
        return params


class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = []

        for i in range(len(nouts)):
            i_nonlin = (i != (len(nouts) - 1))  # last layer is linear
            self.layers.append(
                Layer(sz[i], sz[i+1], nonlin=i_nonlin)
            )

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x 
    
    def parameters(self):
        params = []
        for layer in self.layers:
            p = layer.parameters()
            params.extend(p)
        return params
    

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def step(self, lr):
        for p in self.parameters():
            p.data -= lr * p.grad
    


def mse_loss(preds, targets):
    # Need to flatten preds if nested
    if isinstance(preds[0], list):
        preds = [p for batch in preds for p in batch]
    
    loss = Value(0.0)
    for p, t in zip(preds, targets):
        t_val = t if isinstance(t, Value) else Value(t)
        loss += (p - t_val) ** 2
    return loss / len(preds)  # Normalize by batch size




