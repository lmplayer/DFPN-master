#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf

import locality_aware_nms as nms_locality
import lanms

tf.app.flags.DEFINE_string('test_data_path', './debug_img/', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', './checkpoints/resnet_v1_50-model.ckpt-0', '')
tf.app.flags.DEFINE_string('output_dir', './results/', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')
tf.app.flags.DEFINE_boolean('select_wh', True, 'True: use w,h; False:use t,r,b,l')
tf.app.flags.DEFINE_boolean('is_select_background', True, 'select_background')
tf.app.flags.DEFINE_float('resize_ratio', 1.0, '')
tf.app.flags.DEFINE_integer('max_side_len', 1280, '')
tf.app.flags.DEFINE_integer('min_side_len', 512, '')
tf.app.flags.DEFINE_float('score_map_thresh', 0.9, '')
tf.app.flags.DEFINE_float('mask_thresh', 0.7, '')
tf.app.flags.DEFINE_float('rl_iou_th', 0.8, 'iou threshold for RegLink')
tf.app.flags.DEFINE_float('min_area', 16*16/4/4, 'filter min area')#16*16/4/4
tf.app.flags.DEFINE_string('gt_path', "./gt_mat", 'gt_path')
tf.app.flags.DEFINE_bool('mask_dilate', True, 'mask_dilate')
tf.app.flags.DEFINE_float('dilate_ratio', 0.1, 'dilate_ratio')
tf.app.flags.DEFINE_string('link_method', "RegLink", '')#"Box", "Mask", "DBSCAN"
#tf.app.flags.DEFINE_integer('gpu_nms_id', -1, '')
tf.app.flags.DEFINE_boolean('use_2branch', True, 'use_FSM')
tf.app.flags.DEFINE_boolean('test_IC15', False, 'test ICDAR 2015')

import model
from icdar import restore_rectangle, parse_mat
from blocks.RegLink import RegLink_func
from sklearn.cluster import DBSCAN

FLAGS = tf.app.flags.FLAGS

GPU_IOU_ID = -1

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
gpuids = FLAGS.gpu_list.split(',')
if len(gpuids) > 2:
    print("please use less than 2 gpus")
    exit()
elif len(gpuids) > 1:
    GPU_IOU_ID = int(gpuids[1])
    if gpuids[1] == gpuids[0]:
        os.environ['CUDA_VISIBLE_DEVICES'] =  gpuids[0]
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    
if FLAGS.link_method == "DBSCAN" and GPU_IOU_ID != -1:
    print("GPU_IOU_ID:",GPU_IOU_ID)
    from polynms.polynms.gpu_iou_matrix import gpu_iou_matrix 


    
ANCHOR_SIZES = [ 32, 64, 128, 256, 512, 1024]
f_fscale_index = [0, 0, 1, 2, 3, 3]
f_scale_i = [1, 1, 2, 4, 8, 8]
if FLAGS.select_wh:
    select_split_N = 2
else:
    select_split_N = 4

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files

def load_gt_totaltext(gt_path, gt_name):
    '''
    load annotation from the text file
    :param gt_path, gt_name:
    :return:
    '''
    text_polys = []
    text_tags = []
    txt_fn = os.path.join(gt_path, 'gt_' + gt_name + '.mat')
    # print(txt_fn)
    if not os.path.exists(txt_fn):
        txt_fn = os.path.join(gt_path, 'poly_gt_' + gt_name + '.mat')
        if not os.path.exists(txt_fn):
            txt_fn = os.path.join(gt_path, gt_name + '.mat')
            if not os.path.exists(txt_fn):
                print('text file {} does not exists'.format(txt_fn))
                return np.array(text_polys, dtype=np.float32)
    return parse_mat(txt_fn)

def resize_image(im, max_side_len=FLAGS.max_side_len, re_ratio=1.):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if re_ratio < 0:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    if max(resize_h * re_ratio, resize_w * re_ratio) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = re_ratio
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)

def resize_image_range(im, min_side_len=FLAGS.min_side_len, max_side_len=FLAGS.max_side_len, re_ratio=FLAGS.resize_ratio):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h
    
    if min(resize_w,resize_h) < min_side_len:
        if resize_w > resize_h:
            resize_w = int(resize_w * float(min_side_len) / resize_h)
            resize_h = min_side_len
        else:
            resize_h = int(resize_h * float(min_side_len) / resize_w)
            resize_w = min_side_len
            
    if max(resize_w,resize_h) > max_side_len:
        if resize_w > resize_h:
            resize_h = int(resize_h * float(max_side_len) / resize_w)
            resize_w = max_side_len
        else:
            resize_w = int(resize_w * float(max_side_len) / resize_h)
            resize_h = max_side_len

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 + 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 + 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=FLAGS.score_map_thresh, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    
    geo_map_idx_ = geo_map[xy_text[:, 0], xy_text[:, 1], :]
    angle = geo_map_idx_[:, 4]
    xy_text_0 = xy_text[angle >= 0]
    xy_text_1 = xy_text[angle < 0]
    xy_text = np.concatenate([xy_text_0, xy_text_1])
    
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer, [], boxes

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return None, timer, [], boxes


def detect_mask(score_map, score_map_full, geo_map, timer, score_map_thresh=FLAGS.score_map_thresh, mask_thresh=FLAGS.mask_thresh, box_thresh=0.1, nms_thres=0.2, min_area=FLAGS.min_area):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param mask_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        score_map_full = score_map_full[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    geo_map_idx_ = geo_map[xy_text[:, 0], xy_text[:, 1], :]
    angle = geo_map_idx_[:, 4]
    xy_text_0 = xy_text[angle >= 0]
    xy_text_1 = xy_text[angle < 0]
    xy_text = np.concatenate([xy_text_0, xy_text_1])
    
    points = list(xy_text)
    for i in range(len(points)):
        points[i] = tuple(points[i])
        
    #points = list(zip(*np.where(score_map > mask_thresh)))
    points_dict = {}
    for i in range(len(points)):
        points_dict[points[i]] = i
    group_mask = dict.fromkeys(points, -1)
    
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    boxes_all = boxes + 0
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer, [], []

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    #dict_box = {}
    #if len(score_map.shape) == 4:
        #score_map = score_map[0, :, :, 0]
        #geo_map = geo_map[0, :, :, ]
    #xy_text = np.argwhere(score_map > mask_thresh)
    #xy_text = xy_text[np.argsort(xy_text[:, 0])]
    
    
    mask_bin = np.zeros([score_map.shape[0], score_map.shape[1]], dtype=np.uint8)
    mask_colors = np.zeros([score_map.shape[0], score_map.shape[1],3], dtype=np.uint8)
    
    if boxes == [] or boxes is None:
    #if boxes.shape[0] < 1:
        return mask_colors, timer, [], []
    boxes_in = boxes + 0
    boxes_in[:, :8] = (boxes_in[:, :8]/4).astype(np.int32)
    h, w = score_map.shape
    boxes_points = []
    cnt=0
    for box in boxes_in:
        b_ps = []
        b = box[:8].reshape((4,2))
        if np.linalg.norm(b[0] - b[1]) <= 1 or np.linalg.norm(b[3]-b[0]) <= 1:
            continue
        xmin = int(max(np.min(b[:,0]),0))
        xmax = int(min(np.max(b[:,0]),w-1))
        ymin = int(max(np.min(b[:,1]),0))
        ymax = int(min(np.max(b[:,1]),h-1))
        #print(ymin,ymax,xmin,xmax)
        
        local_ = score_map_full[ymin:ymax+1, xmin:xmax+1]
        #print("score_map_full",score_map_full.max(),local_.max)
        local_mask = np.zeros_like(local_)
        b[:,0] -= xmin
        b[:,1] -= ymin
        cv2.fillPoly(local_mask, b.astype(np.int32)[np.newaxis, :, :], 1)
        local_ = local_ * local_mask
        #local_th = local_ + 0
        #print("mask_thresh",mask_thresh)
        #local_th[local_th<=mask_thresh] = 1
        #cv2.imwrite("local_"+str(cnt)+".jpg",local_*255)
        #cv2.imwrite("local_"+str(cnt)+"_th.jpg",local_th*255)
        #cnt += 1
        
        ps_idx = np.argwhere(local_ > mask_thresh)
        
        #ps_idx = np.where((xy_text[:,1]>=xmin) & (xy_text[:,1]<=xmax) & (xy_text[:,0]>=ymin) & (xy_text[:,0]<=ymax))[0]
        #for idx in ps_idx:
            #b_ps.append([xy_text[idx,1], xy_text[idx,0]])
        for idx in ps_idx:
            b_ps.append([idx[1]+xmin, idx[0]+ymin])
        
        if b_ps == []:
            continue
        boxes_points.append(b_ps)
        
    #print("boxes_points",boxes_points)
    
    
    
    mask_contours = []
    
    for b in boxes_points:
        mask_bin *= 0
        b = np.array(b)
        b = b[:, ::-1]
        b = b.transpose(1, 0)
        b = (b[0], b[1])
        mask_bin[:, :][b] = 255
        mask_colors[:, :, :][b] = 255
        
        area_ = np.sum(mask_bin/255)
        if area_ <min_area or area_ >= h*w*0.99:
            continue
        
        dilate_kernel_size = 3
        if FLAGS.mask_dilate:
            points_in_ = np.argwhere(mask_bin == 255)
            p_in = points_in_[int(len(points_in_)/2)]
            #print("p_in",p_in)
            if tuple(p_in) in points_dict:
                box_ = boxes_all[points_dict[tuple(p_in)]]
                poly_h = min(np.linalg.norm(box_[0] - box_[3]), np.linalg.norm(box_[1] - box_[2]))
                poly_w = min(np.linalg.norm(box_[0] - box_[1]), np.linalg.norm(box_[2] - box_[3]))
                dilate_kernel_size = int(min(poly_h, poly_w) * FLAGS.dilate_ratio)
            poly_rect = cv2.minAreaRect(points_in_.astype(np.float32))
            rect_height = min(poly_rect[1][0], poly_rect[1][1]) * FLAGS.dilate_ratio
            dilate_kernel_size = max(int(min(dilate_kernel_size, rect_height)),3)
            #dilate_kernel_size = 3
            #print("dilate_kernel_size",dilate_kernel_size)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
            mask_bin = cv2.dilate(mask_bin, kernel)
        
        contours, hierarchy = cv2.findContours(mask_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        max_contour = contours[0]
        max_area = cv2.contourArea(max_contour)
        for i in range(1,len(contours)):
            if cv2.contourArea(contours[i]) > max_area:
                max_contour = contours[i]
                max_area = cv2.contourArea(max_contour)

        epsilon = 0.01 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        mask_contours.append(approx)

        cv2.drawContours(mask_colors, max_contour, -1, (0, 255, 0), 3)
        cv2.drawContours(mask_colors, approx, -1, (0, 0, 255), 3)
        #cv2.imshow("mask_colors", mask_colors)
        #cv2.waitKey(0)
    return mask_colors, timer, mask_contours, boxes


def calc_iou(poly1, poly2):
    size1 = cv2.contourArea((poly1[0:8].reshape((4,2))).astype(np.float32))
    size2 = cv2.contourArea((poly2[0:8].reshape((4,2))).astype(np.float32))
    inter = cv2.intersectConvexConvex((poly1[0:8].reshape((4,2))).astype(np.float32), (poly2[0:8].reshape((4,2))).astype(np.float32))
    inter_size = inter[0]
    if size1 + size2 - inter_size == 0:
        print("calc_iou error, size1 + size2 - inter_size == 0 !!!!!!!!!!!!")
        return 0
    iou = inter_size / (size1 + size2 - inter_size)
    return iou
def calc_iou_area(poly1, poly2, size1, size2):
    #size1 = cv2.contourArea((poly1[0:8].reshape((4,2))).astype(np.float32))
    #size2 = cv2.contourArea((poly2[0:8].reshape((4,2))).astype(np.float32))
    inter = cv2.intersectConvexConvex((poly1[0:8].reshape((4,2))).astype(np.float32), (poly2[0:8].reshape((4,2))).astype(np.float32))
    inter_size = inter[0]
    if size1 + size2 - inter_size == 0:
        print("calc_iou error, size1 + size2 - inter_size == 0 !!!!!!!!!!!!")
        return 0
    iou = inter_size / (size1 + size2 - inter_size)
    return iou
def detect_pixellink(score_map, geo_map, timer, mask_thresh=FLAGS.mask_thresh, box_thresh=0.1, nms_thres=0.2, min_area=FLAGS.min_area):
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > mask_thresh)
    
    h, w = np.shape(score_map)
    #xy_text = np.argwhere(score_map > mask_thresh)
    # sort the text boxes via the y axis
    #xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    
    geo_map_idx_ = geo_map[xy_text[:, 0], xy_text[:, 1], :]
    angle = geo_map_idx_[:, 4]
    xy_text_0 = xy_text[angle >= 0]
    xy_text_1 = xy_text[angle < 0]
    xy_text = np.concatenate([xy_text_0, xy_text_1])
    
    #print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    
    
    
    points = list(xy_text)
    for i in range(len(points)):
        points[i] = tuple(points[i])
        
    #points = list(zip(*np.where(score_map > mask_thresh)))
    points_dict = {}
    for i in range(len(points)):
        points_dict[points[i]] = i
    group_mask = dict.fromkeys(points, -1)

    mask_contours = []
    
    mask = RegLink_func(points, points_dict, group_mask, h, w, boxes, rl_iou_th=FLAGS.rl_iou_th)
    
    mask_colors = np.zeros([score_map.shape[0],score_map.shape[1],3],dtype=np.uint8)
    mask_bin = np.zeros([score_map.shape[0],score_map.shape[1]],dtype=np.uint8)
    #for i in np.unique(mask):
    
    areas_all = (geo_map[:,:,0] + geo_map[:,:,2]) * (geo_map[:,:,1] + geo_map[:,:,3])
    
    for i in range(1,mask.max()+1):
        mask_bin *= 0
        mask_bin[mask==i] = 255
        area_ = np.sum(mask_bin/255)
        
        empty_ratio = 4*4 * 8
        box_area = areas_all[mask==i].sum() / area_
        #print(areas_all[mask==i])
        #print(box_area , area_)
        if area_ <min_area or area_ >= h*w*0.99 or area_ * empty_ratio < box_area:
            continue
        
        dilate_kernel_size = 3
        if FLAGS.mask_dilate:
            points_in_ = np.argwhere(mask_bin == 255)
            p_in = points_in_[int(len(points_in_)/2)]
            #print("p_in",p_in)
            if tuple(p_in) in points_dict:
                box_ = boxes[points_dict[tuple(p_in)]]
                poly_h = min(np.linalg.norm(box_[0] - box_[3]), np.linalg.norm(box_[1] - box_[2]))
                poly_w = min(np.linalg.norm(box_[0] - box_[1]), np.linalg.norm(box_[2] - box_[3]))
                dilate_kernel_size = int(min(poly_h, poly_w) * FLAGS.dilate_ratio)
                #print("111dilate_kernel_size",dilate_kernel_size)
            poly_rect = cv2.minAreaRect(points_in_.astype(np.float32))
            rect_height = min(poly_rect[1][0], poly_rect[1][1]) * FLAGS.dilate_ratio
            #print("rect_height",rect_height)
            dilate_kernel_size = max(int(min(dilate_kernel_size, rect_height)),3)
            #dilate_kernel_size = 3
            #print("222dilate_kernel_size",dilate_kernel_size)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
            mask_bin = cv2.dilate(mask_bin, kernel)
        
        contours, hierarchy = cv2.findContours(mask_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        epsilon = 0.01 * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True)
        points = approx.reshape((-1, 2))
        
        avgscore, maxscore, danum = box_score_fast(score_map, points)
        #print(avgscore, maxscore)
        #print(danum, area_)
        if 0.6 > avgscore or maxscore < 0.90 or danum < area_ * 0.3:
            #print('score', score)
            continue
            
        points = points.reshape((-1,1,2))
        _, sside = get_mini_boxes(points)
        if sside < 4:
            #print('sside', sside)
            continue

        mask_contours.append(approx)
        
        if not FLAGS.no_write_images:
            mask_colors[mask==i,:] = np.random.randint(100,255,size=3)
    #boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), 0.2)
    timer['restore'] = time.time() - start
   

    return mask_colors, timer, mask_contours, []
    
def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    avg_score = cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]
    max_score = np.max(bitmap[ymin:ymax+1, xmin:xmax+1] * mask)
    danum = len(np.where((bitmap[ymin:ymax+1, xmin:xmax+1] * mask) > 0.85)[0])
    return avg_score, max_score, danum
    
def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])

def detect_dbscan(score_map, geo_map, timer, score_map_thresh=FLAGS.mask_thresh, box_thresh=0.1, nms_thres=0.2, min_area=FLAGS.min_area, gpu_iou_id=GPU_IOU_ID):
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    
    h, w = np.shape(score_map)
    #xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    #xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    
    geo_map_idx_ = geo_map[xy_text[:, 0], xy_text[:, 1], :]
    angle = geo_map_idx_[:, 4]
    xy_text_0 = xy_text[angle >= 0]
    xy_text_1 = xy_text[angle < 0]
    xy_text = np.concatenate([xy_text_0, xy_text_1])
    
    #print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    
    if boxes.shape[0] <= 1:
        return None, timer, [], []
    
    points = list(xy_text)
    for i in range(len(points)):
        points[i] = tuple(points[i])
        
    #points = list(zip(*np.where(score_map > mask_thresh)))
    points_dict = {}
    for i in range(len(points)):
        points_dict[points[i]] = i
    
    print("gpu_iou_id",gpu_iou_id)
    
    if gpu_iou_id < 0:
        print("using cpu IoU")
        time_iou = time.time()
        iou_dict = np.ones((boxes.shape[0],boxes.shape[0]), dtype=np.float32)
        areas_ = np.zeros((boxes.shape[0]), dtype=np.float32)
        for i in range(areas_.shape[0]):
            areas_[i] = cv2.contourArea((boxes[i,0:8].reshape((4,2))).astype(np.float32))
            
        for i in range(iou_dict.shape[0]):
            for j in range(i+1, iou_dict.shape[1]):
                iou_dict[i,j] = calc_iou_area(boxes[i], boxes[j], areas_[i], areas_[j])
        
        iou_dict = 1.0 - iou_dict * iou_dict.T
        print("time_cpu_iou:",time.time() - time_iou)
    else:
        print("using gpu IoU")
        #boxes_iou = boxes[:,:8]
        #boxes_iou = np.array(boxes_iou, dtype=np.float32)
        boxes_iou = []
        for b in boxes:
            boxes_iou.append(cv2.convexHull(b[:8].reshape(4,2),clockwise=False,returnPoints=True).reshape(8))
        boxes_iou=np.array(boxes_iou).astype(np.int32).astype(np.float32)
        
        print(boxes_iou.shape)
        time_iou = time.time()
        iou_dict = 1.0 - gpu_iou_matrix(boxes_iou, boxes_iou, gpu_iou_id)
        print("time_gpu_iou:",time.time() - time_iou)
    #print(iou_dict,iou_dict.shape)
    
    in_index = np.arange((boxes.shape[0]))
    in_index = in_index[:,np.newaxis].astype(np.int32)
    
    def distance(a,b):
        #print(a,b)
        return iou_dict[int(a[0]),int(b[0])]
    
    time_DBSCAN = time.time()
    y_pred = DBSCAN(eps = 0.2, min_samples = 10, metric=lambda a, b : distance(a,b)).fit_predict(in_index)
    print("time_DBSCAN:",time.time() - time_DBSCAN)
    
    print("y_pred.shape",y_pred.shape, y_pred)
    print("xy_text.shape",xy_text.shape)
    #print(xy_text[0,0],xy_text[0,1])
    print(np.unique(y_pred))
    box_cnt = np.unique(y_pred)
    boxes_points = []
    for b_idx_ in range(box_cnt.max()+1):
        p_idxs = np.argwhere(y_pred==b_idx_)[:,0]
        #print("p_idxs.shape,p_idxs:",p_idxs.shape,p_idxs)
        if p_idxs.shape[0] < min_area:
            continue
        b_ps = []
        for p_idx_ in p_idxs:
            b_ps.append([xy_text[p_idx_,1], xy_text[p_idx_,0]])
        boxes_points.append(b_ps)
    
    mask_contours = []
    
    mask_colors = np.zeros([score_map.shape[0],score_map.shape[1],3],dtype=np.uint8)
    mask_bin = np.zeros([score_map.shape[0],score_map.shape[1]],dtype=np.uint8)
    
    for b in boxes_points:
        mask_bin *= 0
        b = np.array(b)
        b = b[:,::-1]
        b = b.transpose(1,0)
        b = (b[0],b[1])
        mask_bin[b] = 255
        area_ = np.sum(mask_bin/255)
        if area_ <min_area or area_ >= h*w*0.99:
            continue
        
        dilate_kernel_size = 3
        if FLAGS.mask_dilate:
            points_in_ = np.argwhere(mask_bin == 255)
            p_in = points_in_[int(len(points_in_)/2)]
            #print("p_in",p_in)
            if tuple(p_in) in points_dict:
                box_ = boxes[points_dict[tuple(p_in)]]
                poly_h = min(np.linalg.norm(box_[0] - box_[3]), np.linalg.norm(box_[1] - box_[2]))
                poly_w = min(np.linalg.norm(box_[0] - box_[1]), np.linalg.norm(box_[2] - box_[3]))
                dilate_kernel_size = int(min(poly_h, poly_w) * FLAGS.dilate_ratio)
            poly_rect = cv2.minAreaRect(points_in_.astype(np.float32))
            rect_height = min(poly_rect[1][0], poly_rect[1][1]) * FLAGS.dilate_ratio
            dilate_kernel_size = max(int(min(dilate_kernel_size, rect_height)),3)
            #dilate_kernel_size = 3
            #print("dilate_kernel_size",dilate_kernel_size)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
            mask_bin = cv2.dilate(mask_bin, kernel)

        contours, hierarchy = cv2.findContours(mask_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        epsilon = 0.01 * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True)
        
        mask_contours.append(approx)
        
        if not FLAGS.no_write_images:
            mask_colors[:,:,:][b] = np.random.randint(100,255,size=3)

    
    timer['restore'] = time.time() - start
   

    return mask_colors, timer, mask_contours, []



def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def main(argv=None):
    FLAGS.output_dir = FLAGS.output_dir + FLAGS.link_method
    try:
        os.makedirs(FLAGS.output_dir)
        os.makedirs(FLAGS.output_dir+"_box")
        os.makedirs(FLAGS.output_dir+"_draw")
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        is_training = tf.placeholder(tf.bool, name='training_flag')
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_geometry, _, f_score_full, f_geometry_full = model.model(input_images, is_training=is_training, anchor_sizes=ANCHOR_SIZES, f_fscale_index=f_fscale_index, f_scale_i=f_scale_i, select_split_N=select_split_N,is_select_background=FLAGS.is_select_background, use_2branch=FLAGS.use_2branch)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
                # ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
                # model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
                model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            else:
                model_path = FLAGS.checkpoint_path
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            durations = 0
            for im_fn in im_fn_list:
                im = cv2.imread(im_fn)[:, :, ::-1]
                start_time = time.time()
                if FLAGS.use_2branch==False:
                    im_resized, (ratio_h, ratio_w) = resize_image(im, max_side_len=FLAGS.max_side_len, re_ratio=FLAGS.resize_ratio)
                else:
                    im_resized, (ratio_h, ratio_w) = resize_image_range(im)
                print(im.shape,im_resized.shape,(ratio_h, ratio_w))

                timer = {'net': 0, 'restore': 0, 'nms': 0}
                start = time.time()
                score, geometry, score_full, geometry_full = sess.run([f_score, f_geometry, f_score_full, f_geometry_full], feed_dict={input_images: [im_resized], is_training: False})
                timer['net'] = time.time() - start
                
                boxes = []
                #mask_contours = []
                if FLAGS.link_method == "Box":
                    mask, timer, mask_contours, boxes = detect(score_map=score, geo_map=geometry, timer=timer)
                elif FLAGS.link_method == "Mask":
                    mask, timer, mask_contours, boxes = detect_mask(score_map=score, score_map_full=score_full, geo_map=geometry, timer=timer)
                elif FLAGS.link_method == "DBSCAN":
                    mask, timer, mask_contours, boxes = detect_dbscan(score_map=score_full, geo_map=geometry_full, timer=timer, gpu_iou_id=GPU_IOU_ID)
                elif FLAGS.link_method == "RegLink":
                    mask, timer, mask_contours, boxes = detect_pixellink(score_map=score_full, geo_map=geometry_full, timer=timer)
                print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                    im_fn, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))
                #print(boxes)
                if boxes is not None and boxes != []:
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h
                duration = time.time() - start_time
                print('[timing] {}'.format(duration))
                durations += duration
                #print(FLAGS.no_write_images)
                if not FLAGS.no_write_images:
                    name = os.path.basename(im_fn).split(".")[0]
                    img_path = os.path.join(FLAGS.output_dir+"_draw", name+"_shrunk.jpg")
                    cv2.imwrite(img_path, score[0]*255)
                    #score_thr = np.where((score_full[0]>0.7), 1, 0)
                    img_path = os.path.join(FLAGS.output_dir+"_draw", name+"_full.jpg")
                    #print('111111', img_path)
                    cv2.imwrite(img_path, score_full[0]*255)
                    img_path = os.path.join(FLAGS.output_dir+"_draw", name+"_mask.jpg")
                    cv2.imwrite(img_path, mask)

                

                # save to file
                if FLAGS.use_2branch==False:
                    res_file = os.path.join(FLAGS.output_dir+"_box", 'res_{}.txt'.format(os.path.basename(im_fn).split('.')[0]))
                    with open(res_file, 'w') as f:
                        if boxes is not None:
                            for box in boxes:
                                # to avoid submitting errors
                                box = sort_poly(box.astype(np.int32))
                                box = cv2.convexHull(box.reshape(4,2),clockwise=False,returnPoints=True)
                                box = box.reshape(4,2)
                                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                    continue
                                f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                                    box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                                ))
                                #f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                                    #box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                                #))
                                cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=3)
                else:
                    res_file = os.path.join(FLAGS.output_dir+"_box", '{}.txt'.format(os.path.basename(im_fn).split('.')[0]))
                    with open(res_file, 'w') as f:
                        if boxes is not None:
                            #k = 0
                            for box in boxes:
                                # to avoid submitting errors
                                box = sort_poly(box.astype(np.int32))
                                box = cv2.convexHull(box.reshape(4,2),clockwise=False,returnPoints=True)
                                box = box.reshape(4,2)
                                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                    continue
                                #f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                                    #box[0, 1], box[0, 0], box[1, 1], box[1, 0], box[2, 1], box[2, 0], box[3, 1], box[3, 0],
                                #))
                                f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                                    box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                                ))
                                #imcopy = im[:, :, ::-1]
                                cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=3)
                                #img_path = os.path.join(FLAGS.output_dir+"_1299boxdraw", str(k) + '.jpg')
                                #print(os.path.basename(im_fn), img_path)
                                #cv2.imwrite(img_path, im[:, :, ::-1])
                                #k += 1
                if mask_contours != []:
                    
                    for ci in range(len(mask_contours)):
                        for p_i in range(mask_contours[ci].shape[0]):
                            mask_contours[ci][p_i,0,0] /= (ratio_w/4.0)
                            mask_contours[ci][p_i,0,1] /= (ratio_h/4.0)
                
                res_file = os.path.join(FLAGS.output_dir, '{}.txt'.format(os.path.basename(im_fn).split('.')[0]))
                with open(res_file, 'w') as f:     
                    for contour in mask_contours:
                        if contour.shape[0] < 3:
                            continue
                        contour_line = ""
                        #for p_i in range(contour.shape[0]-1,0,-1):
                            #contour_line = contour_line + str(contour[p_i,0,0]) + "," + str(contour[p_i,0,1]) + ","
                        #contour_line = contour_line + str(contour[0,0,0]) + "," + str(contour[0,0,1]) + "\n"
                        for p_i in range(contour.shape[0]-1,0,-1):
                            contour_line = contour_line + str(contour[p_i,0,0]) + "," + str(contour[p_i,0,1]) + ","
                        contour_line = contour_line + str(contour[0,0,0]) + "," + str(contour[0,0,1]) + "\n"
                        f.write(contour_line)
                        
                
                if not FLAGS.no_write_images:
                        
                    for contour in mask_contours:
                        cv2.polylines(im[:, :, ::-1], [contour.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=3)
                    
                    #gt_contours, gt_tags = load_gt_totaltext(FLAGS.gt_path, os.path.basename(im_fn).split('.')[0])
                    #for i in range(len(gt_contours)):
                        #if gt_tags[i]:
                            #color = (0, 255, 255)
                        #else:
                            #color = (0, 255, 0)
                        #cv2.polylines(im[:, :, ::-1], [gt_contours[i].astype(np.int32).reshape((-1, 1, 2))], True, color=color, thickness=2)
                        
                    img_path = os.path.join(FLAGS.output_dir+"_draw", os.path.basename(im_fn))
                    cv2.imwrite(img_path, im[:, :, ::-1])
                    
            avg_duration = durations / max(len(im_fn_list),1)
            print(len(im_fn_list))
            print('[avg_timing] {}'.format(avg_duration))

if __name__ == '__main__':
    tf.app.run()
