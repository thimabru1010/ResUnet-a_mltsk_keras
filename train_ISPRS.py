import utils
import time
from utils import np, plt, load_tiff_image, load_SAR_image, compute_metrics, data_augmentation, unet, normalization, \
RGB_image, extract_patches, patch_tiles, bal_aug_patches, extrac_patch2, test_FCN, pred_recostruction, \
weighted_categorical_crossentropy, mask_no_considered, tf, Adam, prediction, load_model, confusion_matrix, \
EarlyStopping, ModelCheckpoint, identity_block, ResNet50, color_map, SGD, \
load_npy_image

from ResUnet_a.model import Resunet_a
#from ResUnet_a.model2 import Resunet_a2
from multitasking_utils import Tanimoto_dual_loss
import argparse
import os

from skimage.util.shape import view_as_windows
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

import gc
import psutil
import ast
from prettytable import PrettyTable
import tensorflow as tf
from tqdm import tqdm
import tensorflow.keras.models as KM

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def extract_patches_test(binary_img_test_ref, patch_size):
    # Extract training patches
    stride = patch_size

    height, width = binary_img_test_ref.shape
    #print(height, width)

    num_patches_h = int(height / stride)
    num_patches_w = int(width / stride)
    #print(num_patches_h, num_patches_w)

    new_shape = (num_patches_h*num_patches_w, patch_size, patch_size)
    new_img_ref = np.zeros(new_shape)
    print(new_img_ref.shape)
    cont = 0
    # rows
    for h in range(num_patches_h):
        #columns
        for w in range(num_patches_w):
            new_img_ref[cont] = binary_img_test_ref[h*stride:(h+1)*stride, w*stride:(w+1)*stride]
            cont += 1
    #print(cont)

    return new_img_ref


def extract_patches_train(img_test_normalized, patch_size):
    # Extract training patches manual
    stride = patch_size

    height, width, channel = img_test_normalized.shape
    #print(height, width)

    num_patches_h = height // stride
    num_patches_w = width // stride
    #print(num_patches_h, num_patches_w)

    new_shape = (num_patches_h*num_patches_w, patch_size, patch_size, channel)
    new_img = np.zeros(new_shape)
    print(new_img.shape)
    cont = 0
    # rows
    for h in range(num_patches_h):
        # columns
        for w in range(num_patches_w):
            new_img[cont] = img_test_normalized[h*stride:(h+1)*stride, w*stride:(w+1)*stride]
            cont += 1
    #print(cont)


    return new_img


def Test(model, patch_test, args):
    result = model.predict(patch_test)
    if args.multitasking:
        predicted_class = np.argmax(result[0], axis=-1)
    else:
        predicted_class = np.argmax(result, axis=-1)
    return predicted_class


def compute_metrics_hw(true_labels, predicted_labels):
    accuracy = 100*accuracy_score(true_labels, predicted_labels)
    #avg_accuracy = 100*accuracy_score(true_labels, predicted_labels, average=None)
    f1score = 100*f1_score(true_labels, predicted_labels, average=None)
    recall = 100*recall_score(true_labels, predicted_labels, average=None)
    precision = 100*precision_score(true_labels, predicted_labels, average=None)
    return accuracy, f1score, recall, precision


def compute_mcc(y_true, y_pred):
    # print('[CHECKING METRICS]')
    # print(len(y_pred))
    # print(len(y_true))
    # print(y_true.shape)
    # print(y_pred.shape)
    true_positives = tf.keras.metrics.TruePositives()
    true_positives.update_state(y_true, y_pred)
    tp = true_positives.result()
    true_negatives = tf.keras.metrics.TrueNegatives()
    true_negatives.update_state(y_true, y_pred)
    tn = true_negatives.result()
    false_positive = tf.keras.metrics.FalsePositives()
    false_positive.update_state(y_true, y_pred)
    fp = false_positive.result()
    false_negative = tf.keras.metrics.FalseNegatives()
    false_negative.update_state(y_true, y_pred)
    fn = false_negative.result()
    mcc = (tp*tn - fp*fn) / tf.math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn+fn))
    return mcc


def compute_metrics_seg(y_true, y_pred):
    print('[CHECKING METRICS]')
    precision = tf.keras.metrics.Precision()
    precision.update_state(y_true, y_pred)
    prec_res = precision.result()
    print(f'precision: {prec_res}')
    recall = tf.keras.metrics.Recall()
    recall.update_state(y_true, y_pred)
    recall_res = recall.result()
    print(f'recall: {recall_res}')
    # mcc = compute_mcc(y_true, y_pred)
    # print(mcc)
    return


def add_tensorboard_scalars(train_writer, val_writer, epoch,
                            metric_name, train_loss, val_loss,
                            train_acc=None, val_acc=None, val_mcc=None):
    with train_writer.as_default():
        tf.summary.scalar(metric_name+'/Loss', train_loss,
                          step=epoch)
        if train_acc is not None:
            tf.summary.scalar(metric_name+'/Accuracy', train_acc,
                              step=epoch)
    with val_writer.as_default():
        tf.summary.scalar(metric_name+'/Loss', val_loss,
                          step=epoch)
        if val_acc is not None:
            tf.summary.scalar(metric_name+'/Accuracy', val_acc,
                              step=epoch)

        if val_mcc is not None:
            tf.summary.scalar(metric_name+'/MCC', val_mcc,
                            step=epoch)


def train_model(args, net, x_train_paths, y_train_paths, x_val_paths,
                y_val_paths, batch_size, epochs, x_shape_batch, y_shape_batch,
                patience=10, delta=0.001):
    # patches_train = x_train_paths
    print('Start training...')
    print('='*60)
    print(f'Training on {len(x_train_paths)} images')
    print(f'Validating on {len(x_val_paths)} images')
    print('='*60)
    print(f'Total Epochs: {epochs}')
    # Initialize tensorboard metrics
    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(args.log_path, 'train'))
    val_summary_writer = tf.summary.create_file_writer(
        os.path.join(args.log_path, 'val'))
    # Initialize as maximum possible number
    min_loss = float('inf')
    cont = 0
    total_train_loss = []
    total_train_acc = []
    total_val_loss = []
    total_val_acc = []
    x_train_b = np.zeros(x_shape_batch, dtype=np.float32)
    y_train_h_b_seg = np.zeros(y_shape_batch, dtype=np.float32)
    x_val_b = np.zeros(x_shape_batch, dtype=np.float32)
    y_val_h_b_seg = np.zeros(y_shape_batch, dtype=np.float32)
    if args.multitasking:
        # Bounds
        if args.bound:
            y_train_h_b_bound = np.zeros(y_shape_batch, dtype=np.float32)
            y_val_h_b_bound = np.zeros(y_shape_batch, dtype=np.float32)
        # Dists
        if args.dist:
            y_train_h_b_dist = np.zeros(y_shape_batch, dtype=np.float32)
            y_val_h_b_dist = np.zeros(y_shape_batch, dtype=np.float32)
        # Colors
        if args.color:
            y_train_h_b_color = np.zeros((y_shape_batch[0],
                                          y_shape_batch[1],
                                          y_shape_batch[2], 3),
                                         dtype=np.float32)
            y_val_h_b_color = np.zeros((y_shape_batch[0],
                                        y_shape_batch[1],
                                        y_shape_batch[2], 3),
                                       dtype=np.float32)
    print(net.metrics_names)
    for epoch in range(epochs):
        if not args.multitasking:
            loss_tr = np.zeros((1, 3), dtype=np.float32)
            loss_val = np.zeros((1, 3), dtype=np.float32)
        else:
            metrics_len = len(net.metrics_names)
            loss_tr = np.zeros((1, metrics_len))
            loss_val = np.zeros((1, metrics_len))
        # Computing the number of batchs on training
        n_batchs_tr = len(x_train_paths)//batch_size
        # Random shuffle the data
        if not args.multitasking:
            (x_train_paths_rand,
             y_train_paths_rand_seg) = shuffle(x_train_paths, y_train_paths[0])
        else:
            (x_train_paths_rand, y_train_paths_rand_seg,
             y_train_paths_rand_bound, y_train_paths_rand_dist,
             y_train_paths_rand_color) \
             = shuffle(x_train_paths, y_train_paths[0], y_train_paths[1],
                       y_train_paths[2], y_train_paths[3])

        # Training the network per batch
        for batch in tqdm(range(n_batchs_tr), desc="Train"):
            x_train_paths_b = x_train_paths_rand[batch * batch_size:(batch + 1) * batch_size]
            y_train_paths_b_seg = y_train_paths_rand_seg[batch * batch_size:(batch + 1) * batch_size]
            if args.multitasking:
                y_train_paths_b_bound = y_train_paths_rand_bound[batch * batch_size:(batch + 1) * batch_size]

                y_train_paths_b_dist = y_train_paths_rand_dist[batch * batch_size:(batch + 1) * batch_size]

                y_train_paths_b_color = y_train_paths_rand_color[batch * batch_size:(batch + 1) * batch_size]
            for b in range(batch_size):
                x_train_b[b] = np.load(x_train_paths_b[b])
                y_train_h_b_seg[b] = np.load(y_train_paths_b_seg[b])
                if args.multitasking:
                    if args.bound:
                        y_train_h_b_bound[b] = np.load(y_train_paths_b_bound[b])
                    if args.dist:
                        y_train_h_b_dist[b] = np.load(y_train_paths_b_dist[b])
                    if args.color:
                        y_train_h_b_color[b] = np.load(y_train_paths_b_color[b])

            if not args.multitasking:
                loss_tr = loss_tr + net.train_on_batch(x_train_b, y_train_h_b_seg)
            else:
                y_train_b = {"seg": y_train_h_b_seg}
                if args.bound:
                    y_train_b['bound'] = y_train_h_b_bound
                if args.dist:
                    y_train_b['dist'] = y_train_h_b_dist
                if args.color:
                    y_train_b['color'] = y_train_h_b_color

                loss_tr = loss_tr + net.train_on_batch(x=x_train_b, y=y_train_b)

            # print('='*30 + ' [CHECKING LOSS] ' + '='*30)
            # print(net.metrics_names)
            # print(type(loss_tr))
            # print(len(loss_tr))
            # print(loss_tr)
            # print(loss_tr.shape)

        # Training loss; Divide by the number of batches
        # print(loss_tr)
        loss_tr = loss_tr/n_batchs_tr

        # Computing the number of batchs on validation
        n_batchs_val = len(x_val_paths)//batch_size

        # Evaluating the model in the validation set
        for batch in tqdm(range(n_batchs_val), desc="Validation"):
            x_val_paths_b = x_val_paths[batch * batch_size:(batch + 1) * batch_size]
            y_val_paths_b_seg = y_val_paths[0][batch * batch_size:(batch + 1) * batch_size]
            if args.multitasking:
                y_val_paths_b_bound = y_val_paths[1][batch * batch_size:(batch + 1) * batch_size]

                y_val_paths_b_dist = y_val_paths[2][batch * batch_size:(batch + 1) * batch_size]

                y_val_paths_b_color = y_val_paths[3][batch * batch_size:(batch + 1) * batch_size]
            for b in range(batch_size):
                x_val_b[b] = np.load(x_val_paths_b[b])
                y_val_h_b_seg[b] = np.load(y_val_paths_b_seg[b])
                if args.multitasking:
                    if args.bound:
                        y_val_h_b_bound[b] = np.load(y_val_paths_b_bound[b])
                    if args.dist:
                        y_val_h_b_dist[b] = np.load(y_val_paths_b_dist[b])
                    if args.color:
                        y_val_h_b_color[b] = np.load(y_val_paths_b_color[b])

            if not args.multitasking:
                loss_val = loss_val + net.test_on_batch(x_val_b, y_val_h_b_seg)
            else:
                # Dict template: y_val_b = {"segmentation": y_val_h_b_seg,
                # "boundary": y_val_h_b_bound, "distance":  y_val_h_b_dist,
                # "color": y_val_h_b_color}
                y_val_b = {"seg": y_val_h_b_seg}
                if args.bound:
                    y_val_b['bound'] = y_val_h_b_bound
                if args.dist:
                    y_val_b['dist'] = y_val_h_b_dist
                if args.color:
                    y_val_b['color'] = y_val_h_b_color

                loss_val = loss_val + net.test_on_batch(x=x_val_b, y=y_val_b)

        loss_val = loss_val/n_batchs_val
        if not args.multitasking:
            print(f'loss_val shape: {loss_val.shape}')
            train_loss = loss_tr[0, 0]
            train_acc = loss_tr[0, 1]
            val_loss = loss_val[0, 0]
            val_acc = loss_val[0, 1]
            val_mcc = loss_val[0, 2]
            total_train_loss.append(train_loss)
            total_train_acc.append(train_acc)
            total_val_loss.append(val_loss)
            total_val_acc.append(val_acc)
            print(f"Epoch: {epoch}" +
                    f"Training loss: {train_loss :.5f}" +
                    f"Train acc.: {100*train_acc:.5f}%" +
                    f"Validation loss: {val_loss :.5f}" +
                    f"Validation acc.: {100*val_acc:.5f}%" +
                    f"Validation mcc.: {val_mcc:.5f}%")

            add_tensorboard_scalars(train_summary_writer, val_summary_writer,
                                    epoch, 'Total', train_loss, val_loss,
                                    train_acc, val_acc, val_mcc)
        else:
            train_metrics = dict(zip(net.metrics_names, loss_tr.tolist()[0]))
            val_metrics = dict(zip(net.metrics_names, loss_val.tolist()[0]))

            metrics_table = PrettyTable()
            metrics_table.title = f'Epoch: {epoch}'
            metrics_table.field_names = ['Task', 'Loss', 'Val Loss',
                                         'Acc %', 'Val Acc %']
            metrics_table.add_row(['Seg', round(train_metrics['seg_loss'], 5),
                                  round(val_metrics['seg_loss'], 5),
                                  round(100*train_metrics['seg_accuracy'], 5),
                                  round(100*val_metrics['seg_accuracy'], 5)])

            add_tensorboard_scalars(train_summary_writer, val_summary_writer,
                                    epoch, 'Segmentation',
                                    train_metrics['seg_loss'],
                                    val_metrics['seg_loss'],
                                    train_metrics['seg_accuracy'],
                                    val_metrics['seg_accuracy'],
                                    val_mcc=val_metrics['seg_compute_mcc'])

            if args.bound:
                metrics_table.add_row(['Bound',
                                       round(train_metrics['bound_loss'], 5),
                                      round(val_metrics['bound_loss'], 5),
                                      0, 0])

                add_tensorboard_scalars(train_summary_writer,
                                        val_summary_writer,
                                        epoch, 'Boundary',
                                        train_metrics['bound_loss'],
                                        val_metrics['bound_loss'])
            if args.dist:
                metrics_table.add_row(['Dist',
                                       round(train_metrics['dist_loss'], 5),
                                       round(val_metrics['dist_loss'], 5),
                                       0, 0])

                add_tensorboard_scalars(train_summary_writer,
                                        val_summary_writer,
                                        epoch, 'Distance',
                                        train_metrics['dist_loss'],
                                        val_metrics['dist_loss'])
            if args.color:
                metrics_table.add_row(['Color',
                                       round(train_metrics['color_loss'], 5),
                                       round(val_metrics['color_loss'], 5),
                                       0, 0])

                add_tensorboard_scalars(train_summary_writer,
                                        val_summary_writer,
                                        epoch, 'Color',
                                        train_metrics['color_loss'],
                                        val_metrics['color_loss'])

            metrics_table.add_row(['Total', round(train_metrics['loss'], 5),
                                  round(val_metrics['loss'], 5),
                                  0, 0])

            add_tensorboard_scalars(train_summary_writer,
                                    val_summary_writer,
                                    epoch, 'Total',
                                    train_metrics['loss'],
                                    val_metrics['loss'])
            val_loss = val_metrics['loss']
            print(metrics_table)
        # Early stop
        # Save the model when loss is minimum
        # Stop the training if the loss don't decreases after patience epochs
        if val_loss >= min_loss + delta:
            cont += 1
            print(f'EarlyStopping counter: {cont} out of {patience}')
            if cont >= patience:
                print("Early Stopping! \t Training Stopped")
                print("Saving model...")
                net.save('weights/model_early_stopping.h5')
                return total_train_loss, total_train_acc,
                total_val_loss, total_val_acc
        else:
            cont = 0
            # best_score = score
            min_loss = val_loss
            print("Saving best model...")
            net.save('weights/best_model.h5')

# End functions definition -----------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--resunet_a", help="choose resunet-a model or not",
                        type=str2bool, default=False)
    parser.add_argument("--multitasking", help="choose resunet-a multitasking \
                        or not", type=str2bool, default=False)
    parser.add_argument("--bound", help="choose resunet-a boundary task or not",
                        type=str2bool, default=True)
    parser.add_argument("--dist", help="choose resunet-a distance task or not",
                        type=str2bool, default=True)
    parser.add_argument("--color", help="choose resunet-a color task or not",
                        type=str2bool, default=True)
    parser.add_argument("--gpu_parallel",
                        help="choose 1 to train one multiple gpu",
                        type=str2bool, default=False)
    parser.add_argument("--log_path", help="Path where to save logs",
                        type=str, default='./results/log_run1')
    parser.add_argument("--dataset_path", help="Path where to load dataset",
                        type=str, default='./DATASETS/patch_size=256_stride=32')
    parser.add_argument("-bs", "--batch_size", help="Batch size on training",
                        type=int, default=4)
    parser.add_argument("-lr", "--learning_rate",
                        help="Learning rate on training",
                        type=float, default=1e-3)
    parser.add_argument("--loss", help="choose which loss you want to use",
                        type=str, default='weighted_cross_entropy',
                        choices=['weighted_cross_entropy', 'cross_entropy',
                                 'tanimoto'])
    parser.add_argument("-optm", "--optimizer",
                        help="Choose which optmizer to use",
                        type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument("--num_classes", help="Number of classes",
                        type=int, default=5)
    parser.add_argument("--epochs", help="Number of epochs",
                        type=int, default=500)
    parser.add_argument("-ps", "--patch_size", help="Size of patches extracted",
                        type=int, default=256)
    parser.add_argument("--bound_weight", help="Boundary loss weight",
                        type=float, default=1.0)
    parser.add_argument("--dist_weight", help="Distance transform loss weight",
                        type=float, default=1.0)
    parser.add_argument("--color_weight", help="HSV transform loss weight",
                        type=float, default=1.0)
    args = parser.parse_args()

    print('='*30 + 'INITIALIZING' + '='*30)
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f'GPUS DEVICES: {gpu_devices}')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    if args.gpu_parallel:
        strategy = tf.distribute.MirroredStrategy()
        print(f'Number of devices: {strategy.num_replicas_in_sync}')
    else:
        strategy = None

    tf.config.experimental_run_functions_eagerly(False)
    #tf.config.run_functions_eagerly(True)


    # Load images

    root_path = args.dataset_path
    train_path = os.path.join(root_path, 'train')
    patches_tr = [os.path.join(train_path, name)
                  for name in os.listdir(train_path)]

    ref_path = os.path.join(root_path, 'labels/seg')
    patches_tr_lb_h = [os.path.join(ref_path, name) for name
                       in os.listdir(ref_path)]

    if args.multitasking:
        ref_bound_path = os.path.join(root_path, 'labels/bound')
        print(ref_bound_path)
        patches_bound_labels = [os.path.join(ref_bound_path, name) for name
                                in os.listdir(ref_bound_path)]

        ref_dist_path = os.path.join(root_path, 'labels/dist')
        patches_dist_labels = [os.path.join(ref_dist_path, name) for name
                               in os.listdir(ref_dist_path)]

        ref_color_path = os.path.join(root_path, 'labels/color')
        patches_color_labels = [os.path.join(ref_color_path, name) for name
                                in os.listdir(ref_color_path)]

    if args.multitasking:
        patches_tr, patches_val, patches_tr_lb_h, patches_val_lb_h, patches_bound_labels_tr, patches_bound_labels_val, patches_dist_labels_tr, patches_dist_labels_val, patches_color_labels_tr, patches_color_labels_val   = train_test_split(patches_tr, patches_tr_lb_h, patches_bound_labels, patches_dist_labels, patches_color_labels,  test_size=0.2, random_state=42)
    else:
        patches_tr, patches_val, patches_tr_lb_h, patches_val_lb_h = train_test_split(patches_tr, patches_tr_lb_h, test_size=0.2, random_state=42)

    if args.multitasking:
        '''
            index maps:
                0 --> segmentation
                1 --> boundary
                2 --> distance
                3 --> color
        '''
        y_paths = [patches_tr_lb_h, patches_bound_labels_tr,
                   patches_dist_labels_tr, patches_color_labels_tr]

        val_paths = [patches_val_lb_h, patches_bound_labels_val,
                     patches_dist_labels_val, patches_color_labels_val]
    else:
        y_paths = [patches_tr_lb_h]

        val_paths = [patches_val_lb_h]

    rows = args.patch_size
    cols = args.patch_size
    channels = 3

    if args.optimizer == 'adam':
        optm = Adam(lr=args.learning_rate, beta_1=0.9)
    elif args.optimizer == 'sgd':
        optm = SGD(lr=args.learning_rate, momentum=0.8)

    print('='*60)
    if args.loss == 'cross_entropy':
        print('Using Cross Entropy')
        loss = "categorical_crossentropy"
        loss_color = "categorical_crossentropy"
    elif args.loss == "tanimoto":
        print('Using Tanimoto Dual Loss')
        loss = Tanimoto_dual_loss()
        loss_color = Tanimoto_dual_loss()
    else:
        print('Using Weighted cross entropy')
        weights = [4.34558461, 2.97682037, 3.92124661, 5.67350328, 374.0300152]
        print(weights)
        loss = weighted_categorical_crossentropy(weights)
        loss_color = "categorical_crossentropy"
    print('='*60)

    if args.resunet_a:

        if args.multitasking:
            print('Multitasking enabled!')
            resuneta = Resunet_a((rows, cols, channels), args.num_classes, args)
            if args.gpu_parallel:
                inp_out = resuneta.model
            else:
                model = resuneta.model
                model.summary()

            losses = {'seg': loss}
            lossWeights = {'seg': 1.0}
            if args.bound:
                losses['bound'] = loss
                lossWeights["bound"] = args.bound_weight
            if args.dist:
                losses['dist'] = loss
                lossWeights["dist"] = args.dist_weight
            if args.color:
                losses['color'] = loss_color
                lossWeights["color"] = args.color_weight

            print(f'Loss Weights: {lossWeights}')
            if args.gpu_parallel:
                with strategy.scope():
                    inputs, out = inp_out
                    model = KM.Model(inputs=inputs, outputs=out)
                    model.summary()
                    model.compile(optimizer=optm, loss=losses,
                                  loss_weights=lossWeights,
                                  metrics={'seg': ['accuracy', compute_mcc]})
            else:
                model.compile(optimizer=optm, loss=losses,
                              loss_weights=lossWeights, metrics={'seg': ['accuracy', tf.keras.metrics.TruePositives(),
                                                                         tf.keras.metrics.FalsePositives(),
                                                                         tf.keras.metrics.TrueNegatives(),
                                                                         tf.keras.metrics.FalseNegatives()]})
        else:
            resuneta = Resunet_a((rows, cols, channels), args.num_classes, args)
            model = resuneta.model
            model.summary()
            model.compile(optimizer=optm, loss=loss, metrics=['accuracy', compute_mcc])

        print('ResUnet-a compiled!')
    else:
        model = unet((rows, cols, channels), args.num_classes)
        model.summary()

        model.compile(optimizer=optm, loss=loss, metrics=['accuracy', compute_mcc])

    filepath = './models/'

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # train the model
    if args.multitasking:
        x_shape_batch = (args.batch_size, args.patch_size, args.patch_size, 3)
        y_shape_batch = (args.batch_size, args.patch_size, args.patch_size, 5)
        start_time = time.time()
        train_model(args, model, patches_tr, y_paths, patches_val, val_paths,
                    args.batch_size, args.epochs,
                    x_shape_batch=x_shape_batch, y_shape_batch=y_shape_batch)
        end_time = time.time() - start_time
        print(f'\nTraining took: {end_time / 3600} \n')
    else:
        x_shape_batch = (args.batch_size, args.patch_size, args.patch_size, 3)
        y_shape_batch = (args.batch_size, args.patch_size, args.patch_size, 5)

        start_time = time.time()

        train_model(args, model, patches_tr, y_paths, patches_val, val_paths,
                    args.batch_size, args.epochs,
                    x_shape_batch=x_shape_batch, y_shape_batch=y_shape_batch)

        end_time = time.time() - start_time
        print(f'\nTraining took: {end_time / 3600} \n')
