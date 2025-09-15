# Convolutional neural networks

[my-website]: https://AJG91.github.io "my-website"
[MNIST-docs]: https://www.openml.org/search?type=data&sort=runs&id=554&status=active "MNIST-docs"

This repository contains code that demonstrates how to create and train a convolutional neural network (CNN) on the [MNIST][MNIST-docs] dataset.

## Getting Started

* This project relies on `python=3.12`. It was not tested with different versions.
  To view the entire list of required packages, see `requirements.txt`.
* Clone the repository to your local machine.
* Once you have `cd` into this repo, create a virtual environment (assuming you have `conda` installed) via
```bash
conda env create -f environment.yml
```
* Enter the virtual environment with `conda activate build-cnn-env`
* Install MedMNIST with `pip install --no-deps medmnist==3.0.1`
* Install the packages in the repo root directory using `pip install -e .`
  (you only need the `-e` option if you intend to edit the source code in `build_cnn/`)


## Example

See [my website][my-website] for examples on how to use this code.
