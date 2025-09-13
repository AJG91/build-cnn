
import numpy as np
import torch as tc
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from typing import Tuple


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
        loader = LoadDatasets(dataset, **kwargs)
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

class LoadDatasets():
    """
    Load and return datasets as PyTorch tensors.

    Attributes
    ----------
    dataset : str
        Name of the dataset to load (e.g., 'MNIST', 'MEDMNIST').
    X : tc.Tensor
        Input features of the dataset as a PyTorch tensor.
        shape: (num_samples, channels, height, width)
    y : tc.Tensor
        Target labels as a PyTorch tensor.

    Parameters
    ----------
    dataset : str
        The name of the dataset to load.
        Options: 'MNIST' or 'MEDMNIST'
    **kwargs
        Additional keyword arguments passed to the specific dataset loader methods.

    Raises
    ------
    ValueError
        If an unsupported dataset name is provided.
    """
    def __init__(self, dataset: str, **kwargs):
        self.dataset = dataset.upper()
        self.loaders = {
            'MNIST': lambda: self.load_mnist(**kwargs),
            'MEDMNIST': lambda: self.load_medmnist(**kwargs),
        }

        if self.dataset not in self.loaders:
            raise ValueError(f'Dataset not supported -> {dataset}. '
                             f'Available: {list(self.loaders.keys())}')

        self.X, self.y = self.loaders[self.dataset]()
        
    def load_mnist(self, version: int = 1) -> Tuple[tc.Tensor, tc.Tensor]:
        """
        Load the MNIST dataset and convert it to PyTorch tensors.

        Parameters
        ----------
        version : int, optional (default=1)
            The version of the MNIST dataset from OpenML.

        Returns
        -------
        X : tc.Tensor
            MNIST images as a float32 tensor normalized to [0, 1].
            Shape: (num_samples, 1, 28, 28)
        y : tc.Tensor
            MNIST labels as int64 tensor.
            Shape: (num_samples,)
        """
        from sklearn.datasets import fetch_openml
        
        X, y = fetch_openml('mnist_784', version=version, return_X_y=True, as_frame=False)
        X = tc.tensor(X, dtype=tc.float32)
        y = tc.from_numpy(y.astype('int64'))
        return X.view(-1, 1, 28, 28), y
        
    def load_medmnist(self, flag: str = 'bloodmnist') -> Tuple[tc.Tensor, tc.Tensor]:
        """
        Load a MedMNIST dataset and convert it to PyTorch tensors.

        Parameters
        ----------
        flag : str, optional (default='bloodmnist')
            Specifies which MedMNIST dataset to load.
            Must be one of the keys in `medmnist.INFO`.

        Returns
        -------
        X : tc.Tensor
            MedMNIST images as a float32 tensor normalized to [0, 1].
            Shape: (num_samples, channels, height, width)
        y : tc.Tensor
            MedMNIST labels as long tensors.
            Shape: (num_samples,)

        Raises
        ------
        ValueError
            If the specified `flag` does not correspond to a valid MedMNIST dataset.
        """
        import medmnist
        from medmnist import INFO
        
        if flag not in INFO:
            raise ValueError(f'Invalid MedMNIST flag -> {flag}. '
                             f'Available datasets: {list(INFO.keys())}"')

        print(INFO[flag]['label'])
        DataClass = getattr(medmnist, INFO[flag]['python_class'])
        
        train_ds = DataClass(split='train', download=True)
        val_ds = DataClass(split='val', download=True)
        test_ds = DataClass(split='test', download=True)
        
        X = np.concatenate([train_ds.imgs, val_ds.imgs, test_ds.imgs], axis=0)
        y = np.concatenate([train_ds.labels, val_ds.labels, test_ds.labels], axis=0)
        
        X = tc.tensor(X, dtype=tc.float32)
        y = tc.tensor(y).long().squeeze()
        return X.permute(0, 3, 1, 2), y


    