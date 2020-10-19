import time
from utils import np, unet, weighted_categorical_crossentropy, Adam, SGD, load_model, K

from ResUnet_a.model2 import Resunet_a
from multitasking_utils import Tanimoto_dual_loss
import argparse
import os

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from prettytable import PrettyTable
import tensorflow as tf
from tqdm import tqdm
import tensorflow.keras.models as KM
import tensorflow.keras as KE


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def compute_mcc(tp, tn, fp, fn):
    mcc = (tp*tn - fp*fn) / tf.math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn+fn))
    return mcc


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
                patience=10, delta=0.001, metrics_names=None):
    # patches_train = x_train_paths
    print('Start training...')
    print('='*60)
    print(f'Training on {len(x_train_paths)} images')
    print(f'Validating on {len(x_val_paths)} images')
    print('='*60)
    print(f'Total Epochs: {epochs}')
    # Initialize tensorboard metrics
    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(args.results_path, 'logs', 'train'))
    val_summary_writer = tf.summary.create_file_writer(
        os.path.join(args.results_path, 'logs', 'val'))
    # Initialize as maximum possible number
    min_loss = float('inf')
    cont = 0
    x_train_b = np.zeros(x_shape_batch, dtype=np.float32)
    y_train_h_b_seg = np.zeros(y_shape_batch, dtype=np.float32)
    x_val_b = np.zeros(x_shape_batch, dtype=np.float32)
    y_val_h_b_seg = np.zeros(y_shape_batch, dtype=np.float32)
    if args.multitasking:
        # Bounds
        y_train_h_b_bound = np.zeros(y_shape_batch, dtype=np.float32)
        y_val_h_b_bound = np.zeros(y_shape_batch, dtype=np.float32)
        # Dists
        y_train_h_b_dist = np.zeros(y_shape_batch, dtype=np.float32)
        y_val_h_b_dist = np.zeros(y_shape_batch, dtype=np.float32)
        # Colors
        y_train_h_b_color = np.zeros((y_shape_batch[0],
                                      y_shape_batch[1],
                                      y_shape_batch[2], 3),
                                     dtype=np.float32)
        y_val_h_b_color = np.zeros((y_shape_batch[0],
                                    y_shape_batch[1],
                                    y_shape_batch[2], 3),
                                   dtype=np.float32)

    # print(net.metrics_names)
    print(net.output_names)
    for epoch in range(epochs):
        # metrics_len = len(net.metrics_names)
        metrics_len = len(metrics_names)
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
            # if args.multitasking:
            #     y_train_paths_b_bound = y_train_paths_rand_bound[batch * batch_size:(batch + 1) * batch_size]
            #     y_train_paths_b_dist = y_train_paths_rand_dist[batch * batch_size:(batch + 1) * batch_size]
            #     y_train_paths_b_color = y_train_paths_rand_color[batch * batch_size:(batch + 1) * batch_size]
            for b in range(batch_size):
                x_train_b[b] = np.load(x_train_paths_b[b])
                y_train_h_b_seg[b] = np.load(y_train_paths_b_seg[b]).astype(np.float32)
                # if args.multitasking:
                #     y_train_h_b_bound[b] = np.load(y_train_paths_b_bound[b])
                #     y_train_h_b_dist[b] = np.load(y_train_paths_b_dist[b])
                #     y_train_h_b_color[b] = np.load(y_train_paths_b_color[b])

            if not args.multitasking:
                loss_tr = loss_tr + net.train_on_batch(x_train_b, y_train_h_b_seg)
            else:
                # Get paths per batch on multitasking labels
                y_train_paths_b_bound = y_train_paths_rand_bound[batch * batch_size:(batch + 1) * batch_size]
                y_train_paths_b_dist = y_train_paths_rand_dist[batch * batch_size:(batch + 1) * batch_size]
                y_train_paths_b_color = y_train_paths_rand_color[batch * batch_size:(batch + 1) * batch_size]
                # Load multitasking labels
                for b in range(batch_size):
                    y_train_h_b_bound[b] = np.load(y_train_paths_b_bound[b]).astype(np.float32)
                    y_train_h_b_dist[b] = np.load(y_train_paths_b_dist[b]).astype(np.float32)
                    y_train_h_b_color[b] = np.load(y_train_paths_b_color[b]).astype(np.float32)

                y_train_b = {"seg": y_train_h_b_seg}
                y_train_b['bound'] = y_train_h_b_bound
                y_train_b['dist'] = y_train_h_b_dist
                y_train_b['color'] = y_train_h_b_color

                loss_tr = loss_tr + net.train_on_batch(x=x_train_b, y=y_train_b, return_dict=False)

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
            # if args.multitasking:
            #     y_val_paths_b_bound = y_val_paths[1][batch * batch_size:(batch + 1) * batch_size]
            #     y_val_paths_b_dist = y_val_paths[2][batch * batch_size:(batch + 1) * batch_size]
            #     y_val_paths_b_color = y_val_paths[3][batch * batch_size:(batch + 1) * batch_size]
            for b in range(batch_size):
                x_val_b[b] = np.load(x_val_paths_b[b])
                y_val_h_b_seg[b] = np.load(y_val_paths_b_seg[b]).astype(np.float32)
                # if args.multitasking:
                #     y_val_h_b_bound[b] = np.load(y_val_paths_b_bound[b])
                #     y_val_h_b_dist[b] = np.load(y_val_paths_b_dist[b])
                #     y_val_h_b_color[b] = np.load(y_val_paths_b_color[b])

            if not args.multitasking:
                loss_val = loss_val + net.test_on_batch(x_val_b, y_val_h_b_seg)
            else:
                # Get paths per batch on multitasking labels
                y_val_paths_b_bound = y_val_paths[1][batch * batch_size:(batch + 1) * batch_size]
                y_val_paths_b_dist = y_val_paths[2][batch * batch_size:(batch + 1) * batch_size]
                y_val_paths_b_color = y_val_paths[3][batch * batch_size:(batch + 1) * batch_size]
                # Load multitasking labels
                for b in range(batch_size):
                    y_val_h_b_bound[b] = np.load(y_val_paths_b_bound[b]).astype(np.float32)
                    y_val_h_b_dist[b] = np.load(y_val_paths_b_dist[b]).astype(np.float32)
                    y_val_h_b_color[b] = np.load(y_val_paths_b_color[b]).astype(np.float32)
                # Dict template: y_val_b = {"segmentation": y_val_h_b_seg,
                # "boundary": y_val_h_b_bound, "distance":  y_val_h_b_dist,
                # "color": y_val_h_b_color}
                y_val_b = {"seg": y_val_h_b_seg}
                y_val_b['bound'] = y_val_h_b_bound
                y_val_b['dist'] = y_val_h_b_dist
                y_val_b['color'] = y_val_h_b_color

                loss_val = loss_val + net.test_on_batch(x=x_val_b, y=y_val_b)
        loss_val = loss_val/n_batchs_val

        # train_metrics = dict(zip(net.metrics_names, loss_tr.tolist()[0]))
        # val_metrics = dict(zip(net.metrics_names, loss_val.tolist()[0]))
        train_metrics = dict(zip(metrics_names, loss_tr.tolist()[0]))
        val_metrics = dict(zip(metrics_names, loss_val.tolist()[0]))
        if not args.multitasking:
            # print(f'loss_val shape: {loss_val.shape}')
            train_loss = train_metrics['loss']
            train_acc = train_metrics['accuracy']
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']

            mcc = compute_mcc(val_metrics['true_positives'],
                              val_metrics['true_negatives'],
                              val_metrics['false_positives'],
                              val_metrics['false_negatives'])

            print(f"Epoch: {epoch} " +
                    f"Training loss: {train_loss :.5f} " +
                    f"Train acc.: {100*train_acc:.5f}% " +
                    f"Validation loss: {val_loss :.5f} " +
                    f"Validation acc.: {100*val_acc:.5f}%")

            add_tensorboard_scalars(train_summary_writer, val_summary_writer,
                                    epoch, 'Total', train_loss, val_loss,
                                    train_acc, val_acc, val_mcc=mcc)
        else:
            mcc = compute_mcc(val_metrics['seg_true_positives'],
                              val_metrics['seg_true_negatives'],
                              val_metrics['seg_false_positives'],
                              val_metrics['seg_false_negatives'])

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
                                    val_mcc=mcc)

            metrics_table.add_row(['Bound',
                                   round(train_metrics['bound_loss'], 5),
                                  round(val_metrics['bound_loss'], 5),
                                  0, 0])
            add_tensorboard_scalars(train_summary_writer,
                                    val_summary_writer,
                                    epoch, 'Boundary',
                                    train_metrics['bound_loss'],
                                    val_metrics['bound_loss'])

            metrics_table.add_row(['Dist',
                                   round(train_metrics['dist_loss'], 5),
                                   round(val_metrics['dist_loss'], 5),
                                   0, 0])
            add_tensorboard_scalars(train_summary_writer,
                                    val_summary_writer,
                                    epoch, 'Distance',
                                    train_metrics['dist_loss'],
                                    val_metrics['dist_loss'])

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
                # print("Saving model...")
                # net.save(os.path.join(args.checkpoint, 'model_early_stopping.h5'))
                return net
        else:
            cont = 0
            min_loss = val_loss
            print("Saving best model...")
            net.save(os.path.join(args.results_path, 'best_model.h5'))

# End functions definition -----------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--resunet_a", help="choose resunet-a model or not",
                        type=str2bool, default=False)
    parser.add_argument("--multitasking", help="choose resunet-a multitasking \
                        or not", type=str2bool, default=False)
    parser.add_argument("--gpu_parallel",
                        help="choose 1 to train one multiple gpu",
                        type=str2bool, default=False)
    parser.add_argument("-rp", "--results_path", help="Path where to save logs and model checkpoint. \
                        Logs and checkpoint will be saved inside this folder.",
                        type=str, default='./results/results_run1')
    parser.add_argument("-cp", "--checkpoint_path", help="Path where to load \
                        model checkpoint to continue training",
                        type=str, default=None)
    parser.add_argument("-dp", "--dataset_path", help="Path where to load dataset",
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
    for device in gpu_devices:
        print(device)
        tf.config.experimental.set_memory_growth(device, True)

    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')

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

    # Define optimizer
    if args.optimizer == 'adam':
        optm = Adam(lr=args.learning_rate, beta_1=0.9)
    elif args.optimizer == 'sgd':
        optm = SGD(lr=args.learning_rate, momentum=0.8)

    # Define Loss
    print('='*60)
    if args.loss == 'cross_entropy':
        print('Using Cross Entropy')
        # loss = "categorical_crossentropy"
        loss = tf.keras.losses.CategoricalCrossentropy()
        loss_reg = tf.keras.losses.MeanSquaredError()
        loss_color = "categorical_crossentropy"
    elif args.loss == "tanimoto":
        print('Using Tanimoto Dual Loss')
        loss = Tanimoto_dual_loss()
        loss_color = Tanimoto_dual_loss()
        loss_reg = Tanimoto_dual_loss()
    else:
        print('Using Weighted cross entropy')
        weights = [4.34558461, 2.97682037, 3.92124661, 5.67350328, 374.0300152]
        print(weights)
        loss = weighted_categorical_crossentropy(weights)
        loss_reg = tf.keras.losses.MeanSquaredError()
        loss_color = "categorical_crossentropy"
    print('='*60)

    # Compile Models
    with strategy.scope():
        if args.checkpoint_path is None:
            if args.resunet_a:
                if args.multitasking:
                    print('Multitasking enabled!')
                    losses = {'seg': loss, 'bound': loss,
                              'dist': loss_reg, 'color': loss_reg}
                    lossWeights = {'seg': 1.0, 'bound': args.bound_weight,
                                   'dist': args.dist_weight, 'color': args.color_weight}

                    print(f'Loss Weights: {lossWeights}')
                    resuneta = Resunet_a((rows, cols, channels), args.num_classes, args)
                    model = resuneta.model
                    model.summary()
                    metrics_dict = {'seg': ['accuracy', tf.keras.metrics.TruePositives(),
                                  tf.keras.metrics.FalsePositives(),
                                  tf.keras.metrics.TrueNegatives(),
                                  tf.keras.metrics.FalseNegatives()]}

                    model.compile(optimizer=optm, loss=losses,
                                  loss_weights=lossWeights, metrics=metrics_dict)
                else:
                    print("Using simple ResUnet-a")
                    resuneta = Resunet_a((rows, cols, channels), args.num_classes, args)
                    model = resuneta.model
                    model.summary()
                    model.compile(optimizer=optm, loss=loss, metrics=['accuracy', tf.keras.metrics.TruePositives(),
                                                               tf.keras.metrics.FalsePositives(),
                                                               tf.keras.metrics.TrueNegatives(),
                                                               tf.keras.metrics.FalseNegatives()])

                print('ResUnet-a compiled!')
            else:
                model = unet((rows, cols, channels), args.num_classes)
                model.summary()
                model.compile(optimizer=optm, loss=loss, metrics=['accuracy', tf.keras.metrics.TruePositives(),
                                                           tf.keras.metrics.FalsePositives(),
                                                           tf.keras.metrics.TrueNegatives(),
                                                           tf.keras.metrics.FalseNegatives()])
        else:
            # load checkpoint compiled
            print(f"[INFO] loading {args.checkpoint_path}...")
            model = load_model(args.checkpoint_path)
            model.summary()

            # update the learning rate
            print(f"[INFO] old learning rate: {K.get_value(model.optimizer.lr)}")
            K.set_value(model.optimizer.lr, args.learning_rate)
            print(f"[INFO] new learning rate: {K.get_value(model.optimizer.lr)}")



    # Create folder for logs and model checkpoint
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    # train the model
    if args.multitasking:
        x_shape_batch = (args.batch_size, args.patch_size, args.patch_size, 3)
        y_shape_batch = (args.batch_size, args.patch_size, args.patch_size, 5)
        start_time = time.time()
        metrics_names = ['loss', 'seg_loss', 'bound_loss', 'dist_loss',
                         'color_loss', 'seg_accuracy', 'seg_true_positives',
                         'seg_false_positives', 'seg_true_negatives',
                         'seg_false_negatives']
        train_model(args, model, patches_tr, y_paths, patches_val, val_paths,
                    args.batch_size, args.epochs,
                    x_shape_batch=x_shape_batch, y_shape_batch=y_shape_batch,
                    metrics_names=metrics_names)
        end_time = time.time() - start_time
        print(f'\nTraining took: {end_time / 3600} \n')
    else:
        x_shape_batch = (args.batch_size, args.patch_size, args.patch_size, 3)
        y_shape_batch = (args.batch_size, args.patch_size, args.patch_size, 5)

        start_time = time.time()
        metrics_names = ['loss', 'accuracy', 'true_positives', 'false_positives',
                         'true_negatives', 'false_negatives']
        train_model(args, model, patches_tr, y_paths, patches_val, val_paths,
                    args.batch_size, args.epochs,
                    x_shape_batch=x_shape_batch, y_shape_batch=y_shape_batch,
                    metrics_names=metrics_names)

        end_time = time.time() - start_time
        print(f'\nTraining took: {end_time / 3600} \n')
