# Label para imagens em HSV: HSV_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
import tensorflow.keras.backend as K
import tensorflow as tf
import cv2

def get_boundary(label, kernel_size = (3,3)):
    # print('[TFMERDA3]'*10)
    # tf.compat.v1.enable_eager_execution()
    # print( tf.executing_eagerly() )
    # #label = tf.convert_to_tensor(label)
    # print(type(label))
    # oi = K.eval(label)
    # print(type(oi))
    # #oi = K.get_value(label)
    # #print(type(label.eval(session=tf.compat.v1.Session())))
    # print(type(label.numpy()))
    tlabel = label.numpy().astype(np.uint8)
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


def get_distance(label):
    tlabel = label.numpy().astype(np.uint8)
    dist = cv2.distanceTransform(tlabel,
                                 cv2.DIST_L2,
                                 0)
    dist = cv2.normalize(dist,
                         dist,
                         0, 1.0,
                         cv2.NORM_MINMAX)
    return dist


def multitasking_weighted_categorical_crossentropy(weights):
        """
        A weighted version of keras.objectives.categorical_crossentropy

        Variables:
            weights: numpy array of shape (C,) where C is the number of classes

        Usage:
            weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss,optimizer='adam')
        """

        weights = K.variable(weights)

        def loss(y_true, y_pred):
            '''
                y_pred it's the output of the model:
                out = [out_seg, out_bound, out_dist, out_color]
            '''
            print(y_true)
            print(y_true[0])
            print(y_true[0][0])
            print('[TFMERDA4]'*10)
            tf.compat.v1.enable_eager_execution()
            print( tf.executing_eagerly() )
            # Segmentation
            # scale predictions so that the class probas of each sample sum to 1
            y_pred_seg = y_pred[0]
            y_true = y_true
            y_pred_seg /= K.sum(y_pred_seg, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred_seg = K.clip(y_pred_seg, K.epsilon(), 1 - K.epsilon())
            # calc
            loss_seg = y_true * K.log(y_pred_seg) * weights
            loss_seg = -K.sum(loss_seg, -1)

            #  Boundary
            y_pred_bound = y_pred[1]
            y_true_bound = get_boundary(y_true)
            y_pred_bound /= K.sum(y_pred_bound, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred_bound = K.clip(y_pred_bound, K.epsilon(), 1 - K.epsilon())
            # calc
            loss_bound = y_true_bound * K.log(y_pred_bound) * weights
            loss_bound = -K.sum(loss_bound, -1)

            #  Distance
            y_pred_dist = y_pred[2]
            y_true_dist = get_distance(y_true)
            y_pred_dist /= K.sum(y_pred_dist, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred_dist = K.clip(y_pred_dist, K.epsilon(), 1 - K.epsilon())
            # calc
            loss_dist = y_true_dist * K.log(y_pred_dist) * weights
            loss_dist = -K.sum(loss_dist, -1)

            # Color HSV_img
            y_pred_color = y_pred[3]
            y_true_color = cv2.cvtColor(y_true,cv2.COLOR_BGR2HSV)
            y_pred_color /= K.sum(y_pred_color, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred_color = K.clip(y_pred_color, K.epsilon(), 1 - K.epsilon())
            # calc
            loss_color = y_true_color * K.log(y_pred_color) * weights
            loss_color = -K.sum(loss_color, -1)

            loss = loss_seg + loss_bound + loss_dist + loss_color

            return loss
        return loss
