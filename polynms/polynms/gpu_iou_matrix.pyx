import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "gpu_iou_matrix.hpp":
    void _matrix_iou(np.float32_t*, np.float32_t*, int, np.float32_t*, int, int, int)

def gpu_iou_matrix(np.ndarray[np.float32_t, ndim=2] boxes1, np.ndarray[np.float32_t, ndim=2] boxes2,
            np.int32_t device_id=0):
    cdef int boxes1_num = boxes1.shape[0]
    cdef int boxes_dim = boxes1.shape[1]
    cdef int boxes2_num = boxes2.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] \
        matrix = np.zeros((boxes1_num, boxes2_num), dtype=np.float32)
    _matrix_iou(&matrix[0, 0], &boxes1[0, 0], boxes1_num, &boxes2[0, 0], boxes2_num, boxes_dim, device_id)
    return matrix
