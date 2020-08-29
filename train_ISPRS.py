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
            loss_tr = np.zeros((1, 2), dtype=np.float32)
            loss_val = np.zeros((1, 2), dtype=np.float32)
        else:
            metrics_len = len(net.metrics_names)
            loss_tr = np.zeros((1, metrics_len))
            loss_val = np.zeros((1, metrics_len))
        # Computing the number of batchs on training
        # n_batchs_tr = patches_train.shape[0]//batch_size
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
        for batch in range(n_batchs_tr):
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
                # Dict template: y_train_b = {"segmentation": y_train_h_b_seg,
                # "boundary": y_train_h_b_bound, "distance":  y_train_h_b_dist,
                # "color": y_train_h_b_color}
                y_train_b = {"seg": y_train_h_b_seg}
                if args.bound:
                    y_train_b['bound'] = y_train_h_b_bound
                if args.dist:
                    y_train_b['dist'] = y_train_h_b_dist
                if args.color:
                    y_train_b['color'] = y_train_h_b_color

                # return_dict argument not working
                loss_tr = loss_tr + net.train_on_batch(x=x_train_b, y=y_train_b)

            # print('='*30 + ' [CHECKING LOSS] ' + '='*30)
            # print(net.metrics_names)
            # print(type(loss_tr))
            # print(len(loss_tr))
            # print(loss_tr)
            # print(loss_tr.shape)

        # Training loss; Divide for the numberof batches
        print(loss_tr)
        loss_tr = loss_tr/n_batchs_tr

        # Computing the number of batchs on validation
        # n_batchs_val = x_val_paths.shape[0]//batch_size
        n_batchs_val = len(x_val_paths)//batch_size

        '''
            Talvez fosse bom deletar as matrizes :
            x_train_b
            y_train_h_b
            Antes de começar o validation
        '''

        # Evaluating the model in the validation set
        for batch in range(n_batchs_val):
            x_val_paths_b = x_val_paths[batch * batch_size:(batch + 1) * batch_size]
            y_val_paths_b_seg = y_val_paths[0][batch * batch_size:(batch + 1) * batch_size]
            if args.multitasking:
                # y_train_paths_b_bound
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

        # validation loss
        print(loss_val)
        loss_val = loss_val/n_batchs_val
        if not args.multitasking:
            train_loss = loss_tr[0, 0]
            train_acc = loss_tr[0, 1]
            val_loss = loss_val[0, 0]
            val_acc = loss_val[0, 1]
            total_train_loss.append(train_loss)
            total_train_acc.append(train_acc)
            total_val_loss.append(val_loss)
            total_val_acc.append(val_acc)
            print(f"Epoch: {epoch} \t \
                    Training loss: {train_loss :.5f} \t \
                    Train acc.: {100*train_acc:.5f}% \t \
                    Validation loss: {val_loss :.5f} \t \
                    Validation acc.: {100*val_acc:.5f}%")
        else:
            train_metrics = dict(zip(net.metrics_names, loss_tr.tolist()[0]))
            print(loss_tr.tolist()[0])
            val_metrics = dict(zip(net.metrics_names, loss_val.tolist()[0]))
            train_acc = 0.0
            for task in net.metrics_names:
                if 'accuracy' in task:
                    train_acc += train_metrics[task]
            train_acc = train_acc / 4
            # train_acc = (train_seg_acc + train_bound_acc + train_dist_acc
            #              + train_color_loss) / 4
            total_train_acc.append(train_acc)

            # val_loss = loss_val[0, 0]
            # total_val_loss.append(val_loss)

            val_acc = 0.0
            for task in net.metrics_names:
                if 'accuracy' in task:
                    val_acc += val_metrics[task]
            val_acc = val_acc / 4
            # val_acc = (val_seg_acc + val_bound_acc + val_dist_acc
            #            + val_color_acc) / 4
            total_val_acc.append(val_acc)

            metrics_table = PrettyTable()
            metrics_table.title = f'Epoch: {epoch}'
            metrics_table.field_names = ['Task', 'Loss', 'Val Loss',
                                         'Acc %', 'Val Acc %']
            metrics_table.add_row(['Seg', round(train_metrics['seg_loss'], 5),
                                  round(val_metrics['seg_loss'], 5),
                                  round(100*train_metrics['seg_accuracy'], 5),
                                  round(100*val_metrics['seg_accuracy'], 5)])
            if args.bound:
                metrics_table.add_row(['Bound', round(train_metrics['bound_loss'], 5),
                                      round(val_metrics['bound_loss'], 5),
                                      round(100*train_metrics['bound_accuracy'], 5),
                                      round(100*val_metrics['bound_accuracy'], 5)])
            if args.dist:
                metrics_table.add_row(['Dist', round(train_metrics['dist_loss'], 5),
                                      round(val_metrics['dist_loss'], 5),
                                      round(100*train_metrics['dist_accuracy'], 5),
                                      round(100*val_metrics['dist_accuracy'], 5)])
            if args.color:
                metrics_table.add_row(['Color', round(train_metrics['color_loss'], 5),
                                      round(val_metrics['color_loss'], 5),
                                      round(100*train_metrics['color_accuracy'], 5),
                                      round(100*val_metrics['color_accuracy'], 5)])
            metrics_table.add_row(['Total', round(train_metrics['loss'], 5),
                                  round(val_metrics['loss'], 5),
                                  round(100*train_acc, 5),
                                  round(100*val_acc, 5)])
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

    return total_train_loss, total_train_acc, total_val_loss, total_val_acc


# End functions definition -----------------------------------------------------

if __name__ == '__main__':
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--resunet_a", help="choose resunet-a model or not",
                        type=int, default=0)
    parser.add_argument("--multitasking", help="choose resunet-a multitasking \
                        or not", type=int, default=0)
    parser.add_argument("--bound", help="choose resunet-a boundary task or not",
                        type=int, default=1)
    parser.add_argument("--dist", help="choose resunet-a distance task or not",
                        type=int, default=1)
    parser.add_argument("--color", help="choose resunet-a color task or not",
                        type=int, default=1)
    parser.add_argument("--gpu_parallel", help="choose 1 to train one multiple gpu",
                        type=int, default=0)
    args = parser.parse_args()

    if args.gpu_parallel:
        strategy = tf.distribute.MirroredStrategy()
        print(f'Number of devices: {strategy.num_replicas_in_sync}')
    else:
        strategy = None

    # Load images

    root_path = './DATASETS/patches_ps=256_stride=32'
    train_path = os.path.join(root_path, 'train')
    patches_tr = [os.path.join(train_path, name) for name in os.listdir(train_path)]

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

    number_class = 5
    patch_size = 256
    # stride = patch_size // 8
    batch_size = 4
    epochs = 500


    if args.multitasking:
        # y_paths={"segmentation": patches_tr_lb_h, "boundary": patches_bound_labels_tr, "distance":  patches_dist_labels_tr, "color": patches_color_labels_tr}
        '''
            index maps:
                0 --> segmentation
                1 --> boundary
                2 --> distance
                3 --> color
        '''
        y_paths=[patches_tr_lb_h, patches_bound_labels_tr, patches_dist_labels_tr, patches_color_labels_tr]

        # val_paths={"segmentation": patches_val_lb_h, "boundary": patches_bound_labels_val, "distance":  patches_dist_labels_val, "color": patches_color_labels_val}
        val_paths = [patches_val_lb_h, patches_bound_labels_val, patches_dist_labels_val, patches_color_labels_val]
    else:
        # y_paths={"segmentation": patches_tr_lb_h, "boundary": [], "distance":  [], "color": []}
        y_paths = [patches_tr_lb_h]

        # val_paths={"segmentation": patches_val_lb_h, "boundary": [], "distance":  [], "color": []}
        val_paths = [patches_val_lb_h]



    exp = 1
    rows = patch_size
    cols = patch_size
    channels = 3
    lr = 1e-3
    adam = Adam(lr = lr , beta_1=0.9)
    sgd = SGD(lr=lr,momentum=0.8)

    weights = [4.34558461, 2.97682037, 3.92124661, 5.67350328, 374.0300152]
    print('='*60)
    print(weights)
    loss = weighted_categorical_crossentropy(weights)
    if args.multitasking:
        # weighted_cross_entropy = weighted_categorical_crossentropy(weights)
        # cross_entropy = "categorical_crossentropy"
        tanimoto = Tanimoto_dual_loss()

    if args.resunet_a:

        if args.multitasking:
            print('Multitasking enabled!')
            resuneta = Resunet_a((rows, cols, channels), number_class, args)
            model = resuneta.model
            model.summary()
            # losses = {
            # 	"segmentation": weighted_cross_entropy,
            # 	"boundary": weighted_cross_entropy,
            #     "distance": weighted_cross_entropy,
            #     "color": cross_entropy,
            # }
            # losses = {"segmentation": tanimoto,
            #           "boundary": tanimoto,
            #           "distance": tanimoto,
            #           "color": tanimoto}

            losses = {'seg': tanimoto}
            lossWeights = {'seg': 1.0}
            if args.bound:
                losses['bound'] = tanimoto
                lossWeights["bound"] = 1.0
            if args.dist:
                losses['dist'] = tanimoto
                lossWeights["dist"] = 1.0
            if args.color:
                losses['color'] = tanimoto
                lossWeights["color"] = 1.0

            print(losses)
            print(lossWeights)
            if args.gpu_parallel:
                with strategy.scope():
                    model.compile(optimizer=adam, loss=losses,
                                  loss_weights=lossWeights, metrics=['accuracy'])
            else:
                model.compile(optimizer=adam, loss=losses,
                              loss_weights=lossWeights, metrics=['accuracy'])
        else:
            resuneta = Resunet_a((rows, cols, channels), number_class, args)
            model = resuneta.model
            model.summary()
            model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])

        print('ResUnet-a compiled!')
    else:
        model = unet((rows, cols, channels), number_class)
        model.summary()

        model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
        # model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


    filepath = './models/'

    # train the model
    if args.multitasking:
        x_shape_batch = (batch_size, patch_size, patch_size, 3)
        y_shape_batch = (batch_size, patch_size, patch_size, 5)
        start_time = time.time()
        train_model(args, model, patches_tr, y_paths, patches_val, val_paths, batch_size, epochs, x_shape_batch=x_shape_batch, y_shape_batch=y_shape_batch)
        end_time = time.time() - start_time
    else:
        x_shape_batch = (batch_size, patch_size, patch_size, 3)
        y_shape_batch = (batch_size, patch_size, patch_size, 5)

        start_time = time.time()

        # train_model(args, model, patches_train, patches_tr_lb_h, patches_val, patches_val_lb_h, batch_size, epochs, patience=10, delta=0.001, x_shape_batch=x_shape_batch, y_shape_batch=y_shape_batch, seed=seed)
        train_model(args, model, patches_tr, y_paths, patches_val, val_paths, batch_size, epochs, x_shape_batch=x_shape_batch, y_shape_batch=y_shape_batch)

        end_time = time.time() - start_time
        print(f'\nTraining took: {end_time} \n')

    #%% Test model

    # Load images
    img_test_path = 'Image_Test.npy'
    img_test = load_npy_image(os.path.join(root_path, img_test_path)).astype(np.float32)
    # Normalizes the image
    img_test_normalized = normalization(img_test)
    # Transform the image into W x H x C shape
    img_test_normalized = img_test_normalized.transpose((1,2,0))
    print(img_test_normalized.shape)

    # Load reference
    img_test_ref_path = 'Reference_Test.npy'
    img_test_ref = load_npy_image(os.path.join(root_path, img_test_ref_path))
    img_test_ref = img_test_ref.transpose((1,2,0))
    print(img_test_ref.shape)

    # Create binarized matrix
    w = img_test_ref.shape[0]
    h = img_test_ref.shape[1]
    c = img_test_ref.shape[2]
    # binary_img_train_ref = np.zeros((1,w,h))
    binary_img_test_ref = np.full((w, h), -1)
    # Dictionary used in training
    label_dict = {'(255, 255, 255)': 0, '(0, 255, 0)': 1, '(0, 255, 255)': 2, '(0, 0, 255)': 3, '(255, 255, 0)': 4}
    # label = 0
    for i in range(w):
        for j in range(h):
            r = img_test_ref[i][j][0]
            g = img_test_ref[i][j][1]
            b = img_test_ref[i][j][2]
            rgb = (r,g,b)
            rgb_key = str(rgb)
            binary_img_test_ref[i][j] = label_dict[rgb_key]
    print(label_dict)

    # Put the patch size according to you training here
    patches_test = extract_patches_train(img_test_normalized, patch_size)
    patches_test_ref = extract_patches_test(binary_img_test_ref, patch_size)

    #% Load model
    model = load_model(filepath+'unet_exp_'+str(exp)+'.h5', compile=False)
    # Prediction
    # Test the model
    patches_pred = Test(model, patches_test, args)
    print(patches_pred.shape)

    # Metrics
    true_labels = np.reshape(patches_test_ref, (patches_test_ref.shape[0]* patches_test_ref.shape[1]*patches_test_ref.shape[2]))
    predicted_labels = np.reshape(patches_pred, (patches_pred.shape[0]* patches_pred.shape[1]*patches_pred.shape[2]))

    # Metrics
    metrics = compute_metrics(true_labels,predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0,1,2,3,4])

    print('Confusion  matrix \n', cm)
    print()
    print('Accuracy: ', metrics[0])
    print('F1score: ', metrics[1])
    print('Recall: ', metrics[2])
    print('Precision: ', metrics[3])

    def pred_recostruction(patch_size, pred_labels, binary_img_test_ref):
        # Patches Reconstruction
        stride = patch_size

        height, width = binary_img_test_ref.shape

        num_patches_h = height // stride
        num_patches_w = width // stride

        new_shape = (height, width)
        img_reconstructed = np.zeros(new_shape)
        cont = 0
        # rows
        for h in range(num_patches_h):
            # columns
            for w in range(num_patches_w):
                img_reconstructed[h*stride:(h+1)*stride, w*stride:(w+1)*stride] = patches_pred[cont]
                cont += 1
        print('Reconstruction Done!')
        return img_reconstructed


    def reconstruction_rgb_prdiction_patches(img_reconstructed, label_dict):
        reversed_label_dict = {value : key for (key, value) in label_dict.items()}
        print(reversed_label_dict)
        height, width = img_reconstructed.shape
        img_reconstructed_rgb = np.zeros((height,width,3))
        for h in range(height):
            for w in range(width):
                pixel_class = img_reconstructed[h, w]
                img_reconstructed_rgb[h, w, :] = ast.literal_eval(reversed_label_dict[pixel_class])
        print('Conversion to RGB Done!')
        return img_reconstructed_rgb.astype(np.uint8)

    img_reconstructed = pred_recostruction(patch_size, patches_pred, binary_img_test_ref)
    img_reconstructed_rgb = reconstruction_rgb_prdiction_patches(img_reconstructed, label_dict)

    plt.imsave(f'img_reconstructed_rgb_exp{exp}.jpeg', img_reconstructed_rgb)
