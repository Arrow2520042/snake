"""Build script for optional Cython PER acceleration.

Usage:
    python setup_cython_per.py build_ext --inplace

Notes:
    - Works on Linux/Pop!_OS, macOS, and Windows.
    - If Cython is unavailable, falls back to compiling from per_cython_backend.c.
"""

from setuptools import Extension, setup
import numpy as np

try:
    from Cython.Build import cythonize as _cythonize
    _HAS_CYTHON = True
except Exception:
    _cythonize = None
    _HAS_CYTHON = False


MODULE_NAME = 'per_cython_backend'
SOURCE_FILE = f'{MODULE_NAME}.pyx' if _HAS_CYTHON else f'{MODULE_NAME}.c'


extensions = [
    Extension(
        name=MODULE_NAME,
        sources=[SOURCE_FILE],
        include_dirs=[np.get_include()],
    )
]

if _HAS_CYTHON and _cythonize is not None:
    ext_modules = _cythonize(extensions, language_level='3')
else:
    ext_modules = extensions


setup(
    name=MODULE_NAME,
    ext_modules=ext_modules,
)
