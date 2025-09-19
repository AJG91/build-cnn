
import torch as tc
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from typing import Tuple

from load_dataset import LoadDataset


def get_dataloaders(
    dataset: str,
    split_size: float = 0.1,
    batch_size: int = 64,
    seed: int = 42,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and test sets.

    Parameters
    ----------
    dataset : str
        Name of the dataset to load.
    split_size : float, optional (default=0.1)
        The fraction of data that will be split off for the validation/test set.
    batch_size : int, optional (default=64)
        Number of samples per batch.
    seed : int, optional (default=42)
        Random seed for reproducible splitting of the dataset.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    train_loader : DataLoader
        DataLoader for the training set, with shuffling enabled.
    val_loader : DataLoader
        DataLoader for the validation set, without shuffling.
    test_loader : DataLoader
        DataLoader for the test set, without shuffling.
    """
    data = PreprocessDataset(dataset, split_size, rand_seed=seed, **kwargs)
    return (
        DataLoader(TensorDataset(data.X_train, data.y_train), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(data.X_val, data.y_val), batch_size=batch_size),
        DataLoader(TensorDataset(data.X_test, data.y_test), batch_size=batch_size)
    )

class PreprocessDataset(Dataset):
    """
    Preprocess a dataset and provide train, validation, and test splits as PyTorch tensors.

    Attributes
    ----------
    X_train : tc.Tensor
        Training input features.
    y_train : tc.Tensor
        Training labels.
    X_val : tc.Tensor
        Validation input features.
    y_val : tc.Tensor
        Validation labels.
    X_test : tc.Tensor
        Test input features.
    y_test : tc.Tensor
        Test labels.

    Parameters
    ----------
    dataset : str
        Name of the dataset to load.
        Options: 'MNIST', 'MEDMNIST'
    split_size : float, optional (default=0.1)
        The fraction of data that will be split off for the validation/test set.
    rand_seed : int, optional (default=42)
        Random seed for reproducible splitting.
    **kwargs
        Additional keyword arguments.
    """
    def __init__(
        self, 
        dataset: str, 
        split_size: float = 0.1, 
        rand_seed: int = 42, 
        **kwargs
    ):
        loader = LoadDataset(dataset, **kwargs)
        X, y = loader.X / 255.0, loader.y
        
        X_train, X_val, y_train, y_val = self._split_dataset(X, y, split_size, rand_seed)
        X_train, X_test, y_train, y_test = self._split_dataset(X_train, y_train, split_size, rand_seed)

        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test

    def __len__(self) -> int:
        """Return the total number of samples in the dataset (training + validation + test)."""
        return len(self.y_train) + len(self.y_val) + len(self.y_test)

    def __getitem__(self, idx: int) -> Tuple[tc.Tensor, tc.Tensor]:
        """Return the sample `(X[idx], y[idx])` at the given index."""
        X_all = tc.cat([self.X_train, self.X_val, self.X_test], dim=0)
        y_all = tc.cat([self.y_train, self.y_val, self.y_test], dim=0)
        return X_all[idx], y_all[idx]

    def _split_dataset(
        self, 
        X: tc.Tensor, 
        y: tc.Tensor, 
        split_size: float, 
        rand_seed: int
    ) -> Tuple[tc.Tensor, tc.Tensor, tc.Tensor, tc.Tensor]:
        """
        Split the dataset into train/test or train/validation subsets.

        Parameters
        ----------
        X : tc.Tensor
            Input features.
        y : tc.Tensor
            Target labels.
        split_size : float
            The fraction of data that will be split off for the validation/test set.
        rand_seed : int
            Random seed for reproducibility.

        Returns
        -------
        X_train : tc.Tensor
            Features of the first split.
        X_test : tc.Tensor
            Features of the second split.
        y_train : tc.Tensor
            Labels of the first split.
        y_test : tc.Tensor
            Labels of the second split.
        """
        return train_test_split(X, y, test_size=split_size, random_state=rand_seed)


    