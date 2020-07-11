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


def Tanimoto_loss(label,pred):
    square_pred=tf.square(pred)
    square_label=tf.square(label)
    add_squared_label_pred = tf.add(square_pred,square_label)
    # Ver isso aqui
    sum_square=tf.reduce_sum(add_squared_label_pred,axis=-1)

    product=tf.multiply(pred,label)
    sum_product=tf.reduce_sum(product,axis=-1)
    
    denomintor=tf.subtract(sum_square,sum_product)
    loss=tf.divide(sum_product,denomintor)
    loss=tf.reduce_mean(loss)
    return 1.0-loss
def Tanimoto_dual_loss():
    def loss(label,pred):
        loss1=Tanimoto_loss(pred,label)
        pred=tf.subtract(1.0,pred)
        label=tf.subtract(1.0,label)
        loss2=Tanimoto_loss(label,pred)
        loss=(loss1+loss2)/2
        return loss
    return loss
