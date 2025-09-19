
import torch as tc
from typing import Callable

def train_model(
    model: tc.nn.Module,
    loader: tc.utils.data.DataLoader,
    optimizer: tc.optim.Optimizer,
    loss_fn: Callable[[tc.Tensor, tc.Tensor], tc.Tensor],
    device: tc.device,
) -> float:
    """
    Trains a model over one epoch and calculates the accuracy.

    Parameters
    ----------
    model : torch.nn.Module
        The model that will be trained.
    loader : torch.utils.data.DataLoader
        DataLoader for the training set.
    optimizer : torch.optim.Optimizer
        Optimization algorithm for updating model parameters.
    loss_fn : Callable
        Function that will be used to calculate the discrepancy between predictions and targets.
    device : torch.device or str
        Device on which training is performed.
        Options: 'cpu', 'cuda'
        
    Returns
    -------
    float
        Training accuracy.
    """
    model.train()
    correct, total = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        
        preds = tc.softmax(output, dim=0).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        
    return correct / total
