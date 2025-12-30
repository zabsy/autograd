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






class Trainer:
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Track training progress
        self.losses = []
        self.learning_rates = []

    def train(self, inputs, targets, epochs=100, doLog=True, log_interval=10):
        """
        Training steps:
        - Zero gradients
        - Forward pass to compute predictoins and loss
        - Backward pass to compute gradient
        - Update weights with optimizer
        - Update learning rate with scheduler

        inputs: List of inputs
        targets: List of targets for said inputs
        """

        for epoch in range(epochs):

            self.optimizer.zero_grad()

            # Forward pass
            total_loss = Value(0.0)
            for xi, yi in zip(inputs, targets):
                
                pred = self.model(xi)

                yi_val = yi if isinstance(yi, Value) else Value(yi)
                
                # Accumulate error
                total_loss += (pred - yi_val) **2

            # Average the loss across inputs
            total_loss = total_loss / len(inputs)


            # Backward pass
            total_loss.backward()

            # Update weights and lr
            self.optimizer.step()

            self.scheduler.step(epoch)

            # Print progress
            self.losses.append(total_loss.data)
            self.learning_rates.append(self.optimizer.lr)

            if doLog and epoch % log_interval == 0:
                print(f"Epoch {epoch:4d} | Loss: {total_loss.data:.6f} | LR: {self.optimizer.lr:.6f}")

        if doLog:
            final_loss = self.losses[-1]
            print(f"Final Loss: {final_loss:.6f}")
            
            # Show how model does on the training data
            print(f"\nFinal predictions:")
            for xi, yi in zip(inputs, targets):
                pred = self.model(xi)
                pred_val = pred.data if isinstance(pred, Value) else pred
                print(f"  Input: {xi}, Target: {yi:.2f} | Predicted: {pred_val:.4f}")


    
    def plot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss over time
        ax1.plot(self.losses)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.set_yscale('log')  # uncomment to use log scale if graph is goofy
        ax1.grid(True, alpha=0.3)
        
        # Plot learning rate over time
        ax2.plot(self.learning_rates)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# TEMP
def train_model(model, X, y, epochs=100, lr=0.01, momentum=0.9, end_factor=0.1):
    # Putting it all together for a general train function, with the necessary parameters
    
    # Create optimizer and scheduler
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    
    
    scheduler = LinearLR(optimizer, total_epochs=epochs, end_factor=end_factor)
    
    # Create trainer and train
    trainer = Trainer(model, optimizer, scheduler)
    trainer.train(X, y, epochs=epochs)
    
    return trainer


