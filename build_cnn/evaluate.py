
import torch as tc
from typing import Callable


def evaluate_model(    
    model: tc.nn.Module,
    loader: tc.utils.data.DataLoader,
    device: tc.device,
) -> float:
    """
    Evaluates a model and calculates the accuracy.
    Can be used for evaluating both the validation and test set.

    Parameters
    ----------
    model : torch.nn.Module
        The model that will be trained.
    loader : torch.utils.data.DataLoader
        DataLoader for the training set.
    device : torch.device or str
        Device on which training is performed.
        Options: 'cpu', 'cuda'
        
    Returns
    -------
    float
        Validation/test accuracy.
    """
    model.eval()
    correct, total = 0, 0
    with tc.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            preds = tc.softmax(output, dim=0).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
    return correct / total

def find_misclassified_images(model, loader, device):
    """
    Evaluates the model and saves the images that are missclassified.

    Parameters
    ----------
    model : torch.nn.Module
        The model that will be trained.
    loader : torch.utils.data.DataLoader
        DataLoader for the training set.
    device : torch.device or str
        Device on which training is performed.
        Options: 'cpu', 'cuda'
        
    Returns
    -------
    misclassified_images : list[tc.Tensor]
        List of misclassified images.
    misclassified_preds : list[tc.Tensor]
        List of misclassified predictions that corresponds to the misclassified images.
    true_labels : list[tc.Tensor]
        List of true labels that correspond to the misclassified images.
    """
    misclassified_images = []
    misclassified_preds = []
    true_labels = []
    
    model.eval()
    with tc.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            preds = tc.softmax(output, dim=0).argmax(1)
            mismatches = preds != y
            
            if mismatches.any():
                misclassified_images.extend(X[mismatches].cpu())
                misclassified_preds.extend(preds[mismatches].cpu())
                true_labels.extend(y[mismatches].cpu())
                
    return misclassified_images, misclassified_preds, true_labels


    