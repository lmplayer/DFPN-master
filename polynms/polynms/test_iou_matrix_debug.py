import numpy as np
import time
from gpu_iou_matrix import gpu_iou_matrix

if __name__=='__main__':
    dets = np.array([[10, 20, 105, 10, 100, 101, 17, 100], [22, 30, 120, 30, 100, 120, 40, 120]], dtype=np.float32)
    dets1 = np.tile(dets, (2, 1))
    dets2 = np.tile(dets, (2, 1))
    print(dets,dets.shape)
    for i in range(10):
        t =time.time()
        ret = gpu_iou_matrix(dets1, dets1, 0)
        print(time.time()-t)
    print(ret,ret.shape)
