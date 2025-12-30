import math
from autograd import Value
import matplotlib.pyplot as plt



class SGD:
    """
    SGD: Stochastic Gradient Descent

    Added momentum for smoother training

    Without momentum, updating weights is simply: 
    weight = weight - lr * grad

    However, using momentum (momentum=0.9):
    Instead of moving only based on our current gradient, we can accumulate a "velocity" variable that remembers prev gradients
    
    velocity = momentum * old_velocity - lr * grad
    weight = weight + velocity
    
    """

    # Momentum roughly indicates how much to remember previous gradients
    def __init__(self, parameters, lr=0.01, momentum=0.0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum

        # Initialize velocity at 0, for each parameter
        self.velocity = [0.0 for _ in parameters]

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0

    def step(self):
        for i, p in enumerate(self.parameters):
            if self.momentum > 0:
                # Update velocity step
                self.velocity[i] = self.momentum * self.velocity[i] - self.lr * p.grad

                # Move param by velocity
                p.data += self.velocity[i]

            else:
                # In case momentum not used
                p.data -= self.lr * p.grad

class LinearLR:
    """
    LinearLR: Linear learning rate scheduler

    Learning rate decreases linearly from start_lr to end_lr as training progresses
    """

    def __init__(self, optimizer, total_epochs, end_factor=0.1):
        self.optimizer = optimizer
        self.start_lr = optimizer.lr  # Remember initial learning rate
        self.end_lr = self.start_lr * end_factor  # Calculate ending learning rate
        self.total_epochs = total_epochs

    def step(self, epoch):
        
        progress = epoch / self.total_epochs # 0 to 1, normalized over num of epochs
        self.optimizer.lr = self.start_lr - (self.start_lr - self.end_lr) * progress








