import numpy as np
import time
from gpu_polynms import gpu_polynms

if __name__=='__main__':
    dets = np.array([[10, 20, 105, 10, 100, 101, 17, 100], [22, 30, 120, 30, 100, 120, 40, 120]], dtype=np.float32)
    dets = np.tile(dets, (200, 1))
    scores = np.array([0.6, 0.8], dtype=np.float32)
    scores = np.tile(scores, (200))
    print(dets.shape, scores.shape)
    for i in range(100):
        t =time.time()
        ret = gpu_polynms(dets, scores, 0.5, 0)
        print(time.time()-t)
        print(ret)
