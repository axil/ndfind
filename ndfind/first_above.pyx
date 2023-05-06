# distutils: include_dirs = /home/lev/try_ndfind/env/lib/python3.9/site-packages/numpy/core/include
# distutils: define_macros=NPY_NO_DEPRECATED_API=1
# cython: language_level=3
import numpy as np
cimport numpy as np

from cython cimport floating as cfloating

cimport cython

ctypedef fused integer:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t

ctypedef fused integer2:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t

ctypedef fused signedinteger:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t

ctypedef fused signedinteger2:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t

ctypedef fused unsignedinteger:
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t

ctypedef fused unsignedinteger2:
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t

ctypedef fused floating:
    np.float32_t
    np.float64_t
    np.longdouble_t

ctypedef fused int_or_float:
    integer
    floating

ctypedef fused int_or_float2:
    np.int32_t
    np.int64_t
    np.float64_t

ctypedef fused complexfloating:
    np.complex64_t
    np.complex128_t

ctypedef fused inexact:
    floating
    complexfloating

ctypedef fused numeric:
    integer
    floating
    complexfloating


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _same_type_first_above(int_or_float[:] a, int_or_float v):
    """
    Returns an index of the first occurrence of c in a such that c > v
    If v is missing from a, returns len(a).
    """
    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t i

    for i in range(n):
        if a[i] > v:
            return i
    i = -1
    return i

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _float_first_above(int_or_float[:] a, floating[:] v):
    """
    Returns an index of the first occurrence of c in a such that c > v
    If v is missing from a, returns len(a).
    """
    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t i
    cdef floating v0 = v[0]

    for i in range(n):
        if a[i] > v0:
            return i
    i = -1
    return i

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _signed_first_above(signedinteger[:] a, signedinteger2[:] v):
    """
    Two signed ints
    """
    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t i
    cdef signedinteger2 v0 = v[0]

    for i in range(n):
        if a[i] > v0:
            return i
    i = -1
    return i

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _unsigned_first_above(unsignedinteger[:] a, unsignedinteger2[:] v):
    """
    Two unsigned ints
    """
    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t i
    cdef unsignedinteger2 v0 = v[0]

    for i in range(n):
        if a[i] > v0:
            return i
    i = -1
    return i


def _generic_first_above(a, v):
    indices = np.where(a>v)
    if len(indices[0]):
        return indices[0][0]
    else:
        return -1

def both_signed(a, v):
    return np.issubdtype(a.dtype, np.signedinteger) and isinstance(v, np.signedinteger) or \
           np.issubdtype(a.dtype, np.unsignedinteger) and isinstance(v, np.unsignedinteger)

def first_above(a, v, sorted=False, missing=-1, raises=False):
    """
    Returns an index of the first occurrence of c in a such that c > v
    If v is missing from a, returns len(a).
    """
    a = np.asarray(a)
    
    if np.issubdtype(a.dtype, complex) or isinstance(v, complex):
        raise ValueError('Complex numbers are not comparable.')

    if np.issubdtype(a.dtype, bool) or isinstance(v, bool):
        raise ValueError('`bool` type is not supported.')

    if a.ndim != 1:
        raise ValueError(f'`a` is expected to be 1-dimensional, got {a.ndim}-dimensional array instead')

    if len(a) == 0:
        res = -1
    
    elif sorted:
        res = np.searchsorted(a, v, side='right')
        if res == a.shape[0]:
            res = -1
    
    if np.issubdtype(a.dtype, np.number) and isinstance(v, a.dtype.type) and not isinstance(v, np.float16):
        res = _same_type_first_above(a, v)

    elif np.issubdtype(a.dtype, np.floating) or isinstance(v, np.floating):
        if not np.issubdtype(a.dtype, np.float16) and not isinstance(v, np.float16):
            res = _float_first_above(a, np.array([v], dtype=np.float64))
        else: 
            res = _generic_first_above(a, v)
    
    elif np.issubdtype(a.dtype, np.integer) and isinstance(v, np.integer):
        a_signed, v_signed = np.issubdtype(a.dtype, np.signedinteger), isinstance(v, np.signedinteger)
        if a_signed and v_signed:
            res = _signed_first_above(a, np.array([v]))
        elif not a_signed and not v_signed:
            res = _unsigned_first_above(a, np.array([v]))
        # mixed signedness
        elif v_signed:     # a unsigned
            if v < 0:
                res = 0
            else:
                res = _unsigned_first_above(a, np.array([v], dtype=np.uint64))
        else: # v signed, a unsigned
            if isinstance(v, np.uint64) and v > 2**63:
                res = -1
            else:
                res = _signed_first_above(a, np.array([v], dtype=np.int64))
    else:
        res = _generic_first_above(a, v)
    
    # format the result
    if res == -1:
        if raises:
            raise ValueError(f'{v} is not in array')
        else:
            return missing
    else:
        return res

