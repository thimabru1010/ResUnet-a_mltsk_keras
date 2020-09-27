import tensorflow as tf
import cv2
import numpy as np


def get_boundary_label(label, kernel_size=(3, 3)):
    _, _, channel = label.shape
    bounds = np.empty_like(label, dtype=np.float32)
    for c in range(channel):
        tlabel = label.astype(np.uint8)
        # Apply filter per channel
        temp = cv2.Canny(tlabel[:, :, c], 0, 1)
        tlabel = cv2.dilate(temp,
                            cv2.getStructuringElement(
                                cv2.MORPH_CROSS,
                                kernel_size),
                            iterations=1)
        # Convert to be used on training (Need to be float32)
        tlabel = tlabel.astype(np.float32)
        # Normalize between [0, 1]
        tlabel /= 255.
        bounds[:, :, c] = tlabel
    return bounds


def get_distance_label(label):
    label = label.copy()
    dists = np.empty_like(label, dtype=np.float32)
    for channel in range(label.shape[2]):
        patch = label[:, :, channel].astype(np.uint8)
        dist = cv2.distanceTransform(patch, cv2.DIST_L2, 0)
        dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        dists[:, :, channel] = dist

    return dists


def Tanimoto_loss(label, pred):
    """
    Implementation of Tanimoto loss in tensorflow 2.x
    -------------------------------------------------------------------------
    Tanimoto coefficient with dual from: Diakogiannis et al 2019 (https://arxiv.org/abs/1904.00592)
    """
    smooth = 1e-5

    Vli = tf.reduce_mean(tf.reduce_sum(label, axis=[1,2]), axis=0)
    # wli =  1.0/Vli**2 # weighting scheme
    wli = tf.math.reciprocal(Vli**2) # weighting scheme

    # ---------------------This line is taken from niftyNet package --------------
    # ref: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py, lines:170 -- 172
    # First turn inf elements to zero, then replace that with the maximum weight value
    new_weights = tf.where(tf.math.is_inf(wli), tf.zeros_like(wli), wli)
    wli = tf.where(tf.math.is_inf(wli), tf.ones_like(wli) * tf.reduce_max(new_weights), wli)
    # --------------------------------------------------------------------

    # print('[DEBUG LOSS]')
    # print(label.shape)
    # print(pred.shape)

    square_pred = tf.square(pred)
    square_label = tf.square(label)
    add_squared_label_pred = tf.add(square_pred, square_label)
    sum_square = tf.reduce_sum(add_squared_label_pred, axis=[1, 2])
    # print('sum square')
    # print(sum_square.shape)

    product = tf.multiply(pred, label)
    sum_product = tf.reduce_sum(product, axis=[1, 2])
    # print('sum product')
    # print(sum_product.shape)
    sum_product_labels = tf.reduce_sum(tf.multiply(wli, sum_product), axis=-1)
    # print('sum product labels')
    # print(sum_product_labels.shape)

    denomintor = tf.subtract(sum_square, sum_product)
    # print('denominator')
    # print(denomintor.shape)
    denomintor_sum_labels = tf.reduce_sum(tf.multiply(wli, denomintor), axis=-1)
    # print('denominator sum labels')
    # print(denomintor_sum_labels.shape)
    # Add smooth to avoid numerical instability
    loss = tf.divide(sum_product_labels + smooth, denomintor_sum_labels + smooth)
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
    def loss(label, pred):
        loss1 = Tanimoto_loss(pred, label)
        pred = tf.subtract(1.0, pred)
        label = tf.subtract(1.0, label)
        loss2 = Tanimoto_loss(label, pred)
        loss = (loss1+loss2)*0.5
        return 1.0 - loss
    return loss
