# Label para imagens em HSV: HSV_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
import tensorflow.keras.backend as K
import tensorflow as tf
import cv2
import numpy as np

def get_boundary_labels(labels, _kernel_size = (3,3)):

    labels = labels.copy()
    num_patches, h, w, c = labels.shape
    for n in range(num_patches):
        label = labels[n,:,:,:]
        for channel in range(c):
            #print(label)
            label = label.astype(np.uint8)
            #print(label)
            temp = cv2.Canny(label[:,:,channel],0,1)
            label[:,:,channel] = cv2.dilate(temp, cv2.getStructuringElement(cv2.MORPH_CROSS,_kernel_size) ,iterations = 1)

        labels[n,:,:,:] = label

        labels = labels.astype(np.float32)
        labels /= 255.
        return labels

def get_distance_labels(labels):
    labels = labels.copy()
    #print (label.shape)
    dists = np.empty_like(labels,dtype=np.float32)
    num_patches, h, w, c = labels.shape
    # print('='*10 + ' Distance ' + '='*10)
    # print(labels.shape)
    for n in range(num_patches):
        label = labels[n,:,:,:]
        for channel in range(c):
            label = label.astype(np.uint8)
            dist = cv2.distanceTransform(label[:,:,channel], cv2.DIST_L2, 0)
            dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
            dists[n,:,:,channel] = dist

    return dists


# def get_distance(label):
#     tlabel = label.astype(np.uint8)
#     dist = cv2.distanceTransform(tlabel,
#                                  cv2.DIST_L2,
#                                  0)
#     dist = cv2.normalize(dist,
#                          dist,
#                          0, 1.0,
#                          cv2.NORM_MINMAX)
#     return dist

# def get_distance_labels(patches):
#     # Patch size shape: (amount, h, w, c)
#     (amount, h, w, c) = patches.shape
#     dist_patches = np.zeros([amount, h, w])
#     for i in range(amount):
#         # Remember to convert to opencv color format (bgr) img = img[:,:,::-1]
#         grayimg = cv2.cvtColor(patches[i,:,:,::-1], cv2.COLOR_BGR2GRAY)
#         # Convert the output from bgr to rgb
#         dist_patches[i,:,:] = get_distance(grayimg)
#     return dist_patches

def get_color_labels(patches):
    (amount, h, w, c) = patches.shape
    color_patches = np.zeros([amount, h, w, c])
    for i in range(amount):
        # Remember to convert to opencv color format (bgr) img = img[:,:,::-1]
        hsv_patch = cv2.cvtColor(patches[i,:,:,::-1],cv2.COLOR_BGR2HSV)
        # Normalizes the patches. Good for training. Otherwise loss explodes.
        color_patches[i,:,:,:] = cv2.normalize(hsv_patch, hsv_patch, 0, 1.0, cv2.NORM_MINMAX)
        #print(color_patches[i,:,:,:])
    return color_patches


# img = cv2.imread('teste2.jpg')
# print(img.shape)
# img_b = get_boundary(img)
# cv2.imshow('teste', img_b)
# cv2.waitKey(0)
