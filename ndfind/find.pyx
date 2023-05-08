# distutils: include_dirs = /home/lev/try_ndfind/env/lib/python3.9/site-packages/numpy/core/include
# cython: language_level=3
# include_dirs = C:\Users\ASUS\AppData\Local\Programs\Python\Python310\lib\site-packages\numpy\core\include
import sys
import numpy as np
cimport numpy as np

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
def _signed_find_1d(signedinteger[:] a, signedinteger2[:] va):
    """
    Returns an index of the first occurrence of v in a.
    If v is missing from a, returns -1.
    """
    cdef signedinteger2 v = va[0]
    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t i

    cdef Py_ssize_t res = -1
    for i in range(n):
        if a[i] == v:
            res = i
            break
    return res

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _unsigned_find_1d(unsignedinteger[:] a, unsignedinteger2[:] va):
    """
    Returns an index of the first occurrence of v in a.
    If v is missing from a, returns -1.
    """
    cdef unsignedinteger2 v = va[0]
    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t i

    cdef Py_ssize_t res = -1
    for i in range(n):
        if a[i] == v:
            res = i
            break
    return res

def _int_find_1d(a, v):
    """
    Returns an index of the first occurrence of v in a.
    If v is missing from a, returns -1.
    """
    a_signed, v_signed = np.issubdtype(a.dtype, np.signedinteger), isinstance(v, np.signedinteger)
    if a_signed and v_signed:
        return _signed_find_1d(a, v)
    elif not a_signed and not v_signed:
        return _unsigned_find_1d(a, v)
    elif v_signed: # and a is unsigned
        if v < 0:
            return -1
        else:
            return _unsigned_find_1d(a, np.array([v], dtype=np.uint64))
    else: # unsigned a and signed v
        if isinstance(v, np.uint64) and v > 2**63:
            return -1
        else:
            return _signed_find_1d(a, np.array([v], dtype=np.int64))

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _signed_find_2d(signedinteger[:,:] a, signedinteger2[:] va):
    """
    Returns an index of the first occurrence of v in a.
    If v is missing from a, returns -1.
    """
    cdef signedinteger2 v = va[0]
    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t m = a.shape[1]
    cdef Py_ssize_t i, j

    for i in range(n):
        for j in range(m):
            if a[i, j] == v:
                return m*i+j
    return -1

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _unsigned_find_2d(unsignedinteger[:,:] a, unsignedinteger2[:] va):
    """
    Returns an index of the first occurrence of v in a.
    If v is missing from a, returns -1.
    """
    cdef unsignedinteger2 v = va[0]
    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t m = a.shape[1]
    cdef Py_ssize_t i, j

    for i in range(n):
        for j in range(m):
            if a[i, j] == v:
                return m*i+j
    return -1

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _signed_find_nd(a, signedinteger[:] a0, signedinteger2[:] va):
    """
    Returns an index of the first occurrence of v in a.
    If v is missing from a, returns -1.
    """
    cdef signedinteger2 v = va[0]
    cdef Py_ssize_t res = -1
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n
    cdef np.ndarray[signedinteger] ch
    #cdef integer[:] ch
    for i, chunk in enumerate(np.nditer(a, flags=['external_loop'], order='C')):
        ch = chunk
        n = ch.shape[0]
        for j in range(n):
            if ch[j] == v:
                return i*n+j
    return -1

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _unsigned_find_nd(a, unsignedinteger[:] a0, unsignedinteger2[:] va):
    """
    Returns an index of the first occurrence of v in a.
    If v is missing from a, returns -1.
    """
    cdef unsignedinteger2 v = va[0]
    cdef Py_ssize_t res = -1
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n
    cdef np.ndarray[unsignedinteger] ch
    #cdef integer[:] ch
    for i, chunk in enumerate(np.nditer(a, flags=['external_loop'], order='C')):
        ch = chunk
        n = ch.shape[0]
        for j in range(n):
            if ch[j] == v:
                return i*n+j
    return -1

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _float_find_nd(a, int_or_float[:] a0, floating[:] va, floating rtol=1e-05, floating atol=1e-08):
    """
    Returns an index of the first occurrence of v in a.
    If v is missing from a, returns -1.
    """

    cdef floating v = va[0]
    cdef floating delta = atol + rtol*abs(v)
    cdef floating minv = v - delta
    cdef floating maxv = v + delta

    cdef Py_ssize_t res = -1
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n
    cdef np.ndarray[int_or_float] ch
    for i, chunk in enumerate(np.nditer(a, flags=['external_loop'], order='C')):
        ch = chunk
        n = ch.shape[0]
        for j in range(n):
            if minv < ch[j] < maxv:
                return i*n+j
    return -1

#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing.
#def _int_find_nd(a, floating v, floating rtol=1e-05, floating atol=1e-08):
#    """
#    Returns an index of the first occurrence of v in a.
#    If v is missing from a, returns -1.
#    """
#
#    cdef floating delta = atol + rtol*abs(v)
#    cdef floating minv = v - delta
#    cdef floating maxv = v + delta
#
#    cdef Py_ssize_t res = -1
#    cdef Py_ssize_t i, j
#    cdef Py_ssize_t n
#    cdef np.ndarray[integer] ch
#    for i, chunk in enumerate(np.nditer(a, flags=['external_loop'], order='C')):
#        ch = chunk
#        n = ch.shape[0]
#        for j in range(n):
#            if minv < ch[j] < maxv:
#                return i*n+j
#    return -1

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _complex_find_nd(a, numeric[:] a0, double complex v, double rtol=1e-05, double atol=1e-08):
    """
    Returns an index of the first occurrence of v in a.
    If v is missing from a, returns -1.
    """

    cdef Py_ssize_t res = -1
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n
    cdef np.ndarray[numeric] ch
    cdef double delta = atol + rtol*abs(v)
    for i, chunk in enumerate(np.nditer(a, flags=['external_loop'], order='C')):
        ch = chunk
        n = ch.shape[0]
        for j in range(n):
            if abs(ch[j] - v) <= delta:
                return i*n+j
    return -1

#def _int_find_sorted(a, v):
#    a = np.array(a)
#    n = a.shape[0]
#    i = np.searchsorted(a, v)
##    print(i, a[i] if i<n else '')
#    if i == n or a[i] != v:
#        return -1
#    else:
#        return i

def _float_find_sorted(a, v, rtol=1e-05, atol=1e-08):
    a = np.array(a)
    minv = min(v-atol, v*(1-rtol))
    maxv = max(v+atol, v*(1+rtol))
    n = a.shape[0]
    i = np.searchsorted(a, minv)
#    print(i, a[i] if i<n else '')
    if i == n or a[i] > maxv:
        return -1
    else:
        return i

def _generic_find(a, v, sorted=False):
    """
    a ndarray with dtype in (int, bool, string, bytes, datetime64, object)
    v scalar with type in (int, bool, string, bytes, datetime64, object)
    """
    if sorted:
        i = np.searchsorted(a, v)
        if i == a.shape[0] or a[i] != v:
            return -1
        else:
            return i
    else:
        indices = np.where(a==v)
        if len(indices[0]):
            if a.ndim == 1:
                return indices[0][0]
            else:
                return next(zip(*np.where(a==v)))
        else:
            return -1

def _generic_float_find(a, v, sorted=False):
    """
    a ndarray with dtype in (float, int, bool, string, bytes, datetime64, object)
    v is nan, inf or NINF
    """
    if sorted:
        i = np.searchsorted(a, v)
        if i == a.shape[0]:
            return -1
        elif np.isnan(v):
            if np.isnan(a[i]):
                return i
            else:
                return -1
        elif a[i] == v:
            return i
        else:
            return -1
    else:
        if np.isnan(v):
            indices = np.where(np.isnan(a))
        else:
            indices = np.where(a==v)
        if len(indices[0]):
            if a.ndim == 1:
                return indices[0][0]
            else:
                return next(zip(*np.where(a==v)))
        else:
            return -1

def _py_float_find_sorted(a, v, rtol=1e-05, atol=1e-08):
    """
    a ndarray of ints or floats
    v float
    """
    delta = atol + rtol*abs(v)
    minv = v - delta
    maxv = v + delta
    n = a.shape[0]
    i = np.searchsorted(a, minv)
    if i == n or a[i] > maxv:
        return -1
    else:
        return i

def _py_float_find_unsorted(a, v, rtol=1e-05, atol=1e-08):
    """
    a ndarray of ints or floats
    v float
    """
    indices = np.where(np.isclose(a, v, rtol=rtol, atol=atol))
    if len(indices[0]):
        if a.ndim == 1:
            return indices[0][0]
        else:
            return next(zip(*indices))
    else:
        return -1

def _nan_find(a, sorted=False):
    """
    a ndarray with dtype == object
    v is nan
    """
    if sorted:
        raise ValueError('`sorted=True` optimization does not work when v is NaN')
    for i, ai in enumerate(a):
        if isinstance(ai, (float, np.datetime64)) and np.isnan(ai):
            return i
    return -1

if sys.platform == 'win32':
    def is_complex256(a, v):
        return False
else:
    def is_complex256(a, v):
        return np.issubdtype(a.dtype, np.complex256) or isinstance(v, np.complex256)

def find(a, v, rtol=1e-05, atol=1e-08, sorted=False, missing=-1, raises=False):
#    if not isinstance(a, np.ndarray):
#        a = np.array(a)
    a = np.asarray(a)

    if sorted and a.ndim != 1:
        raise ValueError(f'`sorted=True` optimization only works for 1D arrays, a.ndim={a.ndim}')

    res = None
    generic_float_mode = complex_mode = float_mode = int_mode = nan_mode \
        = generic_mode = signed_int_mode = False
    if is_complex256(a, v):
        generic_float_mode = True
        complex_mode = True
    elif np.issubdtype(a.dtype, np.float16) or isinstance(v, np.float16):
        generic_float_mode = True
    elif np.issubdtype(a.dtype, np.complexfloating):
        if not isinstance(v, complex):
            v = complex(v)
        complex_mode = True
    elif isinstance(v, complex):
        complex_mode = True
    elif isinstance(v, np.complexfloating):
        v = v.item()
        complex_mode = True
    elif np.issubdtype(a.dtype, np.floating):
        if not isinstance(v, (float, np.floating)):
            v = float(v)
        float_mode = True
    elif np.issubdtype(a.dtype, np.number) and \
         isinstance(v, (float, np.floating)):
        float_mode = True
    elif np.issubdtype(a.dtype, np.integer):
        if isinstance(v, int):
            v = np.int_(v)
        elif not isinstance(v, np.integer):
            raise ValueError('Incompatible types of `a` (np.array of '
                            f'{a.dtype}) and `v` ({type(v)})')
        a_signed, v_signed = np.issubdtype(a.dtype, np.signedinteger), isinstance(v, np.signedinteger)
        if a_signed and v_signed:
            signed_int_mode = True
        elif not a_signed and not v_signed:
            pass # signed_int_mode = False
        # mixed signedness
        elif v_signed: # and a is unsigned
            if v < 0:
                res = -1
            else:
                v = np.uint64(v)
                # signed_int_mode = False
        else: # unsigned a and signed v
            if isinstance(v, np.uint64) and v > 2**63:
                res = -1
            else:
                v = np.int64(v)
                signed_int_mode = True
        int_mode = True
    elif isinstance(v, (float, np.datetime64)) and np.isnan(v):
        nan_mode = True
    else:
        generic_mode = True
    
    if res is None:
        if complex_mode and sorted:
            raise ValueError('`sorted=True` optimization cannot be used with complex numbers')
        elif generic_float_mode:
            if sorted:
                res = _py_float_find_sorted(a, v, rtol=rtol, atol=atol)
            else:
                res = _py_float_find_unsorted(a, v, rtol=rtol, atol=atol)
        elif complex_mode:
            if np.isfinite(v):
                res = _complex_find_nd(a, np.zeros(1, dtype=a.dtype), v, rtol=rtol, atol=atol)
            else:
                res = _generic_float_find(a, v, sorted=False)
        elif float_mode:
            if np.isfinite(v):
                if sorted:
                    res = _float_find_sorted(a, v, rtol=rtol, atol=atol)
                else:
                    res = _float_find_nd(a, np.zeros(1, dtype=a.dtype), np.array([v]), rtol=rtol, atol=atol)
            else:
                res = _generic_float_find(a, v, sorted=sorted)
        elif int_mode and not sorted:
            if a.ndim == 1:
                if signed_int_mode:
                    res = _signed_find_1d(a, np.array([v]))
                else:
                    res = _unsigned_find_1d(a, np.array([v]))
            elif a.ndim == 2:
                if signed_int_mode:
                    res = _signed_find_2d(a, np.array([v]))
                else:
                    res = _unsigned_find_2d(a, np.array([v]))
            elif signed_int_mode:
                res = _signed_find_nd(a, np.array([v]))
            else:
                res = _unsigned_find_nd(a, np.array([v]))
        elif nan_mode:
            res = _nan_find(a, sorted=sorted)
        else:
            res = _generic_find(a, v, sorted=sorted)
    
    if a.ndim > 1 and not generic_mode and not generic_float_mode and res != -1:
        return np.unravel_index(res, a.shape)
    elif res == -1:
        if raises:
            raise ValueError(f'{v} is not in array')
        else:
            return missing
    else:
        return res

