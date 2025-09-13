
import torch as tc
import matplotlib.pyplot as plt
from typing import Tuple

def plot_images(
    loader: tc.utils.data.DataLoader, 
    path: str, 
    dpi: int, 
    data: str,
    size: list = [2, 5], 
    figsize: Tuple = (14, 6),
    cmap: str = None
) -> None:
    """
    Plots a subset of the dataset images.

    Parameters
    ----------
    loader : tc.utils.data.DataLoader
        DataLoader for the training set.
    path : str
        String to the directory where the plot is saved.
    dpi : int
        Dots per inch.
        A higher dpi results in a sharper image.
    data : str
        The name of the dataset being plotted.
        This is used to label the plot when saving to directory.
    size : list, optional (default=[2, 5])
        Specifies the number of rows and columns.
    figsize : Tuple, optional (default=[14, 6])
        Specifies the figure size.
    cmap : str, optional (default=None)
        Colormap for 2D images.
    """
    images, labels = next(iter(loader))

    fig, axes = plt.subplots(size[0], size[1], figsize=figsize)
    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0)
        if img.shape[-1] == 1:
            img = img.squeeze()
            cmap = 'gray'
            
        ax.imshow(img, cmap=cmap)
        ax.set_title(f"Label: {labels[i].item()}", fontsize=16)
        ax.axis('off')
    plt.show();
    
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)
    fig.savefig(path + f'example_image_{data}.png', bbox_inches='tight', dpi=dpi)

def plot_misclassified_images(
    images: list[tc.Tensor], 
    preds: list[tc.Tensor], 
    true: list[tc.Tensor], 
    path: str, 
    dpi: int, 
    data: str, 
    size: list = [2, 5], 
    figsize: Tuple = (14, 6),
    cmap: str = None
) -> None:
    """
    Plots the misclassified images along with the predicted and the true label.

    Parameters
    ----------
    images : list[tc.Tensor]
        List of misclassified images.
    preds : list[tc.Tensor]
        List of misclassified predictions that corresponds to the misclassified images.
    true : list[tc.Tensor]
        List of true labels that correspond to the misclassified images.
    path : str
        String to the directory where the plot is saved.
    dpi : int
        Dots per inch.
        A higher dpi results in a sharper image.
    data : str
        The name of the dataset being plotted.
        This is used to label the plot when saving to directory.
    size : list, optional (default=[2, 5])
        Specifies the number of rows and columns.
    figsize : Tuple, optional (default=[14, 6])
        Specifies the figure size.
    cmap : str, optional (default=None)
        Colormap for 2D images.
    """
    fig, axes = plt.subplots(size[0], size[1], figsize=figsize)
    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0)
        if img.shape[-1] == 1:
            img = img.squeeze()
            cmap = 'gray'
            
        ax.imshow(img, cmap=cmap)
        ax.set_title(f"Pred: {preds[i].item()}, True: {true[i].item()}", fontsize=16)
        ax.axis('off')
    plt.show();
    
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)
    fig.savefig(path + f'misclassified_images_{data}.png', bbox_inches='tight', dpi=dpi)

    