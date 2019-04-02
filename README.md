Local diffusion implemented using Python.

Installation of PyCUDA (can be on any Mesabi node):
- module load python2
- conda create -n pycuda3 python=3 numpy
- source activate pycuda3
- module load cuda
- pip install pycuda

Usage (must be on a k40 node):
- module load python2
- source activate pycuda3
- module load cuda