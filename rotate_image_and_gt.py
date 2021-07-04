'''
dst_im, dst_rects = rotate_image_and_gt(src_im, src_rects, degree, scale = 1)
'''

import cv2, csv
import time
import os
import numpy as np
import math
CV_PI = 3.14159265
def rotate_a_point(pt, center, degree,movSize=[0, 0], scale=1):
    angle = degree * CV_PI / 180.0
    alpha = math.cos(angle)
    beta = math.sin(angle)
    x = (pt[0] - center[0]) * scale
    y = (pt[1] - center[1]) * scale
    dstPt = [0, 0]
    dstPt[0] = round( x * alpha + y * beta + center[0] + movSize[0])
    dstPt[1] = round(-x * beta + y * alpha + center[1] + movSize[1])
    return dstPt


def rotate_poly(polys, center, degree, movSize=[0, 0], scale=1):
    det_polys = []
    for p in polys:
        det_polys.append(rotate_a_point(p, center, degree, movSize, scale))
    return det_polys


def shift(size, degree):
    angle = degree * CV_PI / 180.0
    width = size[0]
    height = size[1]

    alpha = math.cos(angle)
    beta = math.sin(angle)
    new_width  = (int)(width * math.fabs(alpha) + height * math.fabs(beta))
    new_height = (int)(width * math.fabs(beta) + height * math.fabs(alpha))

    size = [new_width, new_height]
    return size

def rotate_image_and_gt(src_im, src_rects, degree, scale = 1, is_curve=False):
    degree2 = -degree
    W = src_im.shape[1]
    H = src_im.shape[0]
    center = (W // 2, H // 2)
    newSize = shift([W*scale,H*scale], degree2);
    #dst = Mat(newSize, CV_32FC3)
    M = cv2.getRotationMatrix2D(center, degree2, scale)
    M[0, 2] += (int)((newSize[0] - W) / 2);
    M[1, 2] += (int)((newSize[1] - H) / 2);
    dst_im = cv2.warpAffine(src_im, M, (newSize[0], newSize[1]))
    #warpAffine(src_im, dst, M, cvSize(newSize.width, newSize.height), CV_INTER_LINEAR, BORDER_CONSTANT,sc);

    movSize = [int((newSize[0] - W) / 2), int((newSize[1] - H) / 2)]
    dst_rects = []
    if is_curve==False:
        for rs in src_rects:
            dst_rects.append(rotate_poly(rs,center,degree2,movSize,scale))
        dst_rects = np.array(dst_rects, dtype=np.int32)
    else:
        for rs in src_rects:
            dst_rects.append(np.array(rotate_poly(rs,center,degree2,movSize,scale)))
    return dst_im, dst_rects
