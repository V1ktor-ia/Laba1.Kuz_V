def sum_sq_cython(double[:] lst):
    cdef double s = 0.0
    cdef int i
    cdef int n = lst.shape[0]
    for i in range(n):
        s += lst[i] * lst[i]
    return s