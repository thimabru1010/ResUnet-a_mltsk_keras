# Label para imagens em HSV: HSV_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
import tensorflow.keras.backend as KB
import tensorflow as tf
import cv2
import numpy as np

def get_boundary_labels_old(labels, _kernel_size = (3,3)):

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

def get_boundary_labels(labels, _kernel_size = (3,3)):

    labels = labels.copy()
    num_patches, h, w, c = labels.shape
    bounds = np.empty_like(labels,dtype=np.float32)
    for n in range(num_patches):
        label = labels[n,:,:,:]
        for channel in range(c):
            #print(label)
            label = label.astype(np.uint8)
            #print(label)
            #temp = cv2.Canny(label[:,:,channel],0,1)
            bound = cv2.Canny(label[:,:,channel],0,1)
            bound = cv2.dilate(bound, cv2.getStructuringElement(cv2.MORPH_CROSS,_kernel_size) ,iterations = 1)

            # bound = bound.astype(np.float32)
            # bounds /= 255.

            bounds[n,:,:,channel] = bound

    bounds = bounds.astype(np.float32)
    bounds /= 255.
    return bounds

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
    try:
        (amount, h, w, c) = patches.shape
    except:
        (h, w, c) = patches.shape
        amount = 1

    color_patches = np.zeros([amount, h, w, c])
    for i in range(amount):
        # Remember to convert to opencv color format (bgr) img = img[:,:,::-1]
        hsv_patch = cv2.cvtColor(patches[i,:,:,::-1],cv2.COLOR_BGR2HSV)
        # Normalizes the patches. Good for training. Otherwise loss explodes.
        color_patches[i,:,:,:] = cv2.normalize(hsv_patch, hsv_patch, 0, 1.0, cv2.NORM_MINMAX)
        #print(color_patches[i,:,:,:])
    return color_patches


# def Tanimoto_loss(label,pred):
#     square_pred=tf.square(pred)
#     square_label=tf.square(label)
#     add_squared_label_pred = tf.add(square_pred,square_label)
#     # Ver isso aqui
#     sum_square=tf.reduce_sum(add_squared_label_pred,axis=-1)
#
#     product=tf.multiply(pred,label)
#     sum_product=tf.reduce_sum(product,axis=-1)
#
#     denomintor=tf.subtract(sum_square,sum_product)
#     loss=tf.divide(sum_product,denomintor)
#     loss=tf.reduce_mean(loss)
#     return 1.0-loss


def Tanimoto_loss(label,pred):
    """
    Implementation of Tanimoto loss in tensorflow 2.x
    -------------------------------------------------------------------------
    Tanimoto coefficient with dual from: Diakogiannis et al 2019 (https://arxiv.org/abs/1904.00592)
    """
    smooth = 1e-5

    Vli = tf.reduce_mean(tf.reduce_sum(label,axis=[1,2]),axis=0)
    #wli =  1.0/Vli**2 # weighting scheme
    wli = tf.math.reciprocal(Vli**2) # weighting scheme

    # ---------------------This line is taken from niftyNet package --------------
    # ref: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py, lines:170 -- 172
    # First turn inf elements to zero, then replace that with the maximum weight value
    new_weights = tf.where(tf.is_inf(wli), tf.zeros_like(wli), wli)
    wli = tf.where(tf.is_inf(wli), tf.ones_like(wli) * tf.reduce_max(new_weights), wli)
    # --------------------------------------------------------------------

    # print('[DEBUG LOSS]')
    # print(label.shape)
    # print(pred.shape)

    square_pred=tf.square(pred)
    square_label=tf.square(label)
    add_squared_label_pred = tf.add(square_pred,square_label)
    sum_square=tf.reduce_sum(add_squared_label_pred,axis=[1,2])
    #print('sum square')
    #print(sum_square.shape)

    product=tf.multiply(pred,label)
    sum_product=tf.reduce_sum(product,axis=[1,2])
    # print('sum product')
    # print(sum_product.shape)
    sum_product_labels = tf.reduce_sum(tf.multiply(wli, sum_product), axis=-1)
    # print('sum product labels')
    # print(sum_product_labels.shape)

    denomintor=tf.subtract(sum_square,sum_product)
    # print('denominator')
    # print(denomintor.shape)
    denomintor_sum_labels = tf.reduce_sum(tf.multiply(wli, denomintor), axis=-1)
    # print('denominator sum labels')
    # print(denomintor_sum_labels.shape)
    # Add smooth to avoid numerical instability
    loss=tf.divide(sum_product_labels + smooth,denomintor_sum_labels + smooth)
    # print('loss')
    # print(loss.shape)
    return loss

def Tanimoto_dual_loss():
    '''
        Implementation of Tanimoto dual loss in tensorflow 2.x
        ------------------------------------------------------------------------
            Note: to use it in deep learning training use: return 1. - 0.5*(loss1+loss2)
            OBS: Do use note's advice. Otherwise tanimoto doesn't work
    '''
    def loss(label,pred):
        loss1=Tanimoto_loss(pred,label)
        pred=tf.subtract(1.0,pred)
        label=tf.subtract(1.0,label)
        loss2=Tanimoto_loss(label,pred)
        loss=(loss1+loss2)*0.5
        return 1.0 - loss
    return loss
