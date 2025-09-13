from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='build_cnn',
    version='0.1',
    packages=find_packages(where='build_cnn'),
    package_dir={'': 'build_cnn'},
    py_modules=[splitext(basename(path))[0] for path in glob('build_cnn/*.py')],
    description='Uses PyTorch to train a CNN on various datasets.',
    author='Alberto J. Garcia',
    zip_safe=False
)
