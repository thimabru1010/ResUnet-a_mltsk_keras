# Label para imagens em HSV: HSV_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
import tensorflow.keras.backend as K
import tensorflow as tf
import cv2
import numpy as np

def get_boundary(label, kernel_size = (3,3)):
    #tlabel = label.numpy().astype(np.uint8)
    tlabel = label.astype(np.uint8)
    temp = cv2.Canny(tlabel,0,1)
    tlabel = cv2.dilate(
              temp,
              cv2.getStructuringElement(
              cv2.MORPH_CROSS,
              kernel_size),
              iterations = 1)
    tlabel = tlabel.astype(np.float32)
    tlabel /= 255.
    return tlabel

def get_boundary_labels(patches):
    # Patch size shape: (amount, h, w, c)
    (amount, h, w, c) = patches.shape
    bound_patches = np.zeros([amount, h, w])
    for i in range(amount):
        # Remember to convert to opencv color format (bgr) img = img[:,:,::-1]
        bound_patches[i,:,:] = get_boundary(patches[i,:,:,::-1])
        # cv2.imshow('test', bound_patches[i,:,:])
        # cv2.waitKey(0)
    return bound_patches


def get_distance(label):
    tlabel = label.astype(np.uint8)
    dist = cv2.distanceTransform(tlabel,
                                 cv2.DIST_L2,
                                 0)
    dist = cv2.normalize(dist,
                         dist,
                         0, 1.0,
                         cv2.NORM_MINMAX)
    return dist

def get_distance_labels(patches):
    # Patch size shape: (amount, h, w, c)
    (amount, h, w, c) = patches.shape
    bound_patches = np.zeros([amount, h, w])
    for i in range(amount):
        # Remember to convert to opencv color format (bgr) img = img[:,:,::-1]
        grayimg = cv2.cvtColor(patches[i,:,:,::-1], cv2.COLOR_BGR2GRAY)
        bound_patches[i,:,:] = get_distance(grayimg)
    return bound_patches


# img = cv2.imread('teste2.jpg')
# print(img.shape)
# img_b = get_boundary(img)
# cv2.imshow('teste', img_b)
# cv2.waitKey(0)
