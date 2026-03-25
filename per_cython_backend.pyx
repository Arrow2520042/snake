# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np
cimport numpy as cnp

ctypedef cnp.float64_t f64
ctypedef cnp.int64_t i64


def sample_indices(cnp.ndarray[f64, ndim=1] tree, int capacity, cnp.ndarray[f64, ndim=1] values):
    """Vectorized SumTree leaf retrieval for a batch of sampled cumulative values."""
    cdef Py_ssize_t n = values.shape[0]
    cdef cnp.ndarray[i64, ndim=1] out = np.empty(n, dtype=np.int64)
    cdef Py_ssize_t i
    cdef i64 idx
    cdef i64 left
    cdef f64 v
    cdef Py_ssize_t tree_len = tree.shape[0]

    for i in range(n):
        idx = 0
        v = values[i]
        while True:
            left = 2 * idx + 1
            if left >= tree_len:
                break
            if v <= tree[left]:
                idx = left
            else:
                v -= tree[left]
                idx = left + 1
        out[i] = idx

    return out


def batch_update(cnp.ndarray[f64, ndim=1] tree, cnp.ndarray[i64, ndim=1] indices, cnp.ndarray[f64, ndim=1] priorities):
    """Batch SumTree priority updates with upward propagation."""
    cdef Py_ssize_t n = indices.shape[0]
    cdef Py_ssize_t i
    cdef i64 idx
    cdef i64 parent
    cdef f64 change

    for i in range(n):
        idx = indices[i]
        change = priorities[i] - tree[idx]
        tree[idx] = priorities[i]

        while idx > 0:
            parent = (idx - 1) // 2
            tree[parent] += change
            idx = parent

    return None
