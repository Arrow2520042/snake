"""Build script for optional Cython PER acceleration.

Usage:
    C:/Users/qwert/AppData/Local/Programs/Python/Python314/python.exe setup_cython_per.py build_ext --inplace
"""

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension(
        name="per_cython_backend",
        sources=["per_cython_backend.pyx"],
        include_dirs=[np.get_include()],
    )
]


setup(
    name="per_cython_backend",
    ext_modules=cythonize(extensions, language_level="3"),
)
