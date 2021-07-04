import cv2
import numpy as np
PIXEL_NEIGHBOUR_TYPE_4 = 'PIXEL_NEIGHBOUR_TYPE_4'
PIXEL_NEIGHBOUR_TYPE_8 = 'PIXEL_NEIGHBOUR_TYPE_8'


def get_neighbours_8(x, y):
    """
    Get 8 neighbours of point(x, y)
    """
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), \
        (x - 1, y),                 (x + 1, y),  \
        (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]

def get_neighbours_4(x, y):
    return [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]

def get_neighbours(x, y, neighbour_type=PIXEL_NEIGHBOUR_TYPE_8):
    if neighbour_type == PIXEL_NEIGHBOUR_TYPE_4:
        return get_neighbours_4(x, y)
    else:
        return get_neighbours_8(x, y)
    
def get_neighbours_fn(neighbour_type=PIXEL_NEIGHBOUR_TYPE_8):
    if neighbour_type == PIXEL_NEIGHBOUR_TYPE_4:
        return get_neighbours_4, 4
    else:
        return get_neighbours_8, 8

def is_valid_cord(x, y, w, h):
    """
    Tell whether the 2D coordinate (x, y) is valid or not.
    If valid, it should be on an h x w image
    """
    return x >=0 and x < w and y >= 0 and y < h;

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
def RegLink_func(points, points_dict, group_mask, h, w, boxes, rl_iou_th):
    def find_parent(point):
        return group_mask[point]
        
    def set_parent(point, parent):
        group_mask[point] = parent
        
    def is_root(point):
        return find_parent(point) == -1
    
    def find_root(point):
        root = point
        update_parent = False
        while not is_root(root):
            root = find_parent(root)
            update_parent = True
        
        # for acceleration of find_root
        if update_parent:
            set_parent(point, root)
            
        return root
        
    def join(p1, p2):
        root1 = find_root(p1)
        root2 = find_root(p2)
        
        if root1 != root2:
            set_parent(root1, root2)
        
    def get_all():
        root_map = {}
        def get_index(root):
            if root not in root_map:
                root_map[root] = len(root_map) + 1
            return root_map[root]
        
        mask = np.zeros((h, w), dtype = np.int32)
        for point in points:
            point_root = find_root(point)
            bbox_idx = get_index(point_root)
            mask[point] = bbox_idx
        return mask
    
    # join by link
    #for point in points:
    for p_i in range(len(points)):
        #print(boxes[p_i][-1])
        if boxes[p_i][-1] > 0.6:
            y, x = points[p_i]
            #d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = geo_map[y, x]
            #area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
            neighbours = get_neighbours(x, y)
            for n_idx, (nx, ny) in enumerate(neighbours):
                if is_valid_cord(nx, ny, w, h):
                    if (ny, nx) in points_dict:
                    #d1_pred, d2_pred, d3_pred, d4_pred, theta_pred  = geo_map[ny, nx]
                    #area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
                    #w_union = min(d2_gt, d2_pred) + min(d4_gt, d4_pred)
                    #h_union = min(d1_gt, d1_pred) + min(d3_gt, d3_pred)
                    #area_intersect = w_union * h_union
                    #area_union = area_gt + area_pred - area_intersect
                    #iou_ = ((area_intersect + 1.0)/(area_union + 1.0))
                    #if iou_ > rl_iou_th and abs(theta_gt-theta_pred)<pl_theta_th:
                    #    join(points[p_i], (ny, nx))
                    
                    #print("------------")
                    #print(iou_dict[p_i, points_dict[(ny, nx)]])
                    #print(points[p_i],(ny, nx))
                    #print(p_i, points_dict[(ny, nx)])
                    #print(iou_dict[p_i, points_dict[(ny, nx)]],iou_dict[points_dict[(ny, nx)], p_i])
                        iou_ = calc_iou(boxes[p_i], boxes[points_dict[(ny, nx)]])
                        if iou_ > rl_iou_th:
                            #print(boxes[p_i][-1], boxes[points_dict[(ny, nx)]][-1], iou_)
                            join(points[p_i], (ny, nx))
                #reversed_neighbours = get_neighbours(nx, ny)
                #reversed_idx = reversed_neighbours.index((x, y))
                #link_value = link_mask[y, x, n_idx]# and link_mask[ny, nx, reversed_idx]
                #pixel_cls = pixel_mask[ny, nx]
                #if link_value and pixel_cls:
                    #join(point, (ny, nx))
    
    mask = get_all()
    # print(mask.shape)
    # print(mask.max())
    # cv2.imshow('mask',mask*255)
    # cv2.waitKey(0)
    return mask
