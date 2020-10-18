import utils
import time
from utils import np, plt, load_tiff_image, load_SAR_image, compute_metrics, data_augmentation, unet, normalization, \
RGB_image, extract_patches, patch_tiles, bal_aug_patches, extrac_patch2, test_FCN, pred_recostruction, \
weighted_categorical_crossentropy, mask_no_considered, tf, Adam, prediction, load_model, confusion_matrix, \
EarlyStopping, ModelCheckpoint, identity_block, ResNet50, color_map, SGD, \
load_npy_image
from multitasking_utils import get_boundary_label, get_distance_label

import argparse
import os

from skimage.util.shape import view_as_windows
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

import ast
import cv2
from matplotlib import cm
from matplotlib.colors import hsv_to_rgb
from sklearn.preprocessing import StandardScaler

from multitasking_utils import Tanimoto_dual_loss

def Test(model, patches, args):
    num_patches, weight, height, _ = patches.shape
    preds = model.predict(patches, batch_size=1)
    if args.use_multitasking:
        print('Multitasking Enabled!')
        return preds
    else:
        print(preds.shape)
        predicted_class = np.argmax(preds, axis=-1)
        print(predicted_class.shape)
        return predicted_class


def compute_metrics_hw(true_labels, predicted_labels):
    accuracy = 100*accuracy_score(true_labels, predicted_labels)
    # avg_accuracy = 100*accuracy_score(true_labels, predicted_labels, average=None)
    f1score = 100*f1_score(true_labels, predicted_labels, average=None)
    recall = 100*recall_score(true_labels, predicted_labels, average=None)
    precision = 100*precision_score(true_labels, predicted_labels, average=None)
    return accuracy, f1score, recall, precision


def pred_recostruction(patch_size, pred_labels, binary_img_test_ref, img_type=1):
    # Patches Reconstruction
    if img_type == 1:
        stride = patch_size

        height, width = binary_img_test_ref.shape

        num_patches_h = height // stride
        num_patches_w = width // stride
        #print(num_patches_h, num_patches_w)

        new_shape = (height, width)
        img_reconstructed = np.zeros(new_shape)
        cont = 0
        # rows
        for h in range(num_patches_h):
            # columns
            for w in range(num_patches_w):
                img_reconstructed[h*stride:(h+1)*stride, w*stride:(w+1)*stride] = pred_labels[cont]
                cont += 1
        print('Reconstruction Done!')
    if img_type == 2:
        stride = patch_size

        height, width = binary_img_test_ref.shape

        num_patches_h = height // stride
        num_patches_w = width // stride

        new_shape = (height, width, 3)
        img_reconstructed = np.zeros(new_shape)
        cont = 0
        # rows
        for h in range(num_patches_h):
            # columns
            for w in range(num_patches_w):
                img_reconstructed[h*stride:(h+1)*stride, w*stride:(w+1)*stride, :] = pred_labels[cont]
                cont += 1
        print('Reconstruction Done!')
    return img_reconstructed

def convert_preds2rgb(img_reconstructed, label_dict):
    reversed_label_dict = {value:key for (key, value) in label_dict.items()}
    print(reversed_label_dict)
    height, width = img_reconstructed.shape
    img_reconstructed_rgb = np.zeros((height, width, 3))
    for h in range(height):
        for w in range(width):
            pixel_class = img_reconstructed[h, w]
            img_reconstructed_rgb[h, w, :] = ast.literal_eval(reversed_label_dict[pixel_class])
    print('Conversion to RGB Done!')
    return img_reconstructed_rgb.astype(np.uint8)


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

def colorbar(mappable, ax, fig):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    # ax = mappable.axes
    # fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def convert_hsvpatches2rgb(patches):
    (amount, h, w, c) = patches.shape
    color_patches = np.full([amount, h, w, c], -1)
    for i in range(amount):
        color_patches[i] = hsv_to_rgb(patches[i])*255

    return color_patches


def normalize_rgb(img, norm_type=1):
    # OBS: Images need to be converted to before float32 to be normalized
    # TODO: Maybe should implement normalization with StandardScaler
    # Normalize image between [0, 1]
    if norm_type == 1:
        img /= 255.
    # Normalize image between [-1, 1]
    elif norm_type == 2:
        img /= 127.5 - 1.
    elif norm_type == 3:
        image_reshaped = img.reshape((img.shape[0]*img.shape[1]), img.shape[2])
        scaler = StandardScaler()
        scaler = scaler.fit(image_reshaped)
        image_normalized = scaler.fit_transform(image_reshaped)
        img = image_normalized.reshape(img.shape[0], img.shape[1], img.shape[2])

    return img


def binarize_matrix(img_train_ref, label_dict):
    # Create binarized matrix
    w = img_train_ref.shape[0]
    h = img_train_ref.shape[1]
    # c = img_train_ref.shape[2]
    # binary_img_train_ref = np.zeros((1,w,h))
    binary_img_train_ref = np.full((w, h), -1, dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            r = img_train_ref[i][j][0]
            g = img_train_ref[i][j][1]
            b = img_train_ref[i][j][2]
            rgb = (r, g, b)
            rgb_key = str(rgb)
            binary_img_train_ref[i][j] = label_dict[rgb_key]

    return binary_img_train_ref


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

parser = argparse.ArgumentParser()
parser.add_argument("--use_multitasking",
                    help="Choose resunet-a model or not", action='store_true')
parser.add_argument("--model_path",
                    help="Model's filepath .h5", type=str, required=True)
parser.add_argument("--dataset_path",
                    help="Dataset directory path", type=str, required=True)
parser.add_argument("-ps", "--patch_size",
                    help="Size of Patches extracted from image and reference",
                    type=int, default=256)
parser.add_argument("--norm_type", choices=[1, 2, 3],
                    help="Types of normalization. Be sure to select the same \
                    type used in your training. 1 --> [0,1]; 2 --> [-1,1]; \
                    3 --> StandardScaler() from scikit",
                    type=int, default=1)
parser.add_argument("--num_classes",
                    help="Number of classes",
                    type=int, default=5)
parser.add_argument("--output_path",
                    help="Path to where save predictions",
                    type=str, default='results/preds_run')
args = parser.parse_args()

# Test model

root_path = args.dataset_path

# Load images
img_test_path = 'Image_Test.npy'
img_test = load_npy_image(os.path.join(root_path,
                                       img_test_path)).astype(np.float32)
if args.norm_type == 3:
    img_test_normalized = normalization(img_test)
else:
    img_test_normalized = normalize_rgb(img_test, norm_type=args.norm_type)
# Transform the image into W x H x C shape
img_test_normalized = img_test_normalized.transpose((1, 2, 0))
print(img_test_normalized.shape)

# Load reference
img_test_ref_path = 'Reference_Test.npy'
img_test_ref = load_npy_image(os.path.join(root_path, img_test_ref_path))
# Transform the image into W x H x C shape
img_test_ref = img_test_ref.transpose((1, 2, 0))
print(img_test_ref.shape)

# Dictionary used in training
label_dict = {'(255, 255, 255)': 0, '(0, 255, 0)': 1,
              '(0, 255, 255)': 2, '(0, 0, 255)': 3, '(255, 255, 0)': 4}

binary_img_test_ref = binarize_matrix(img_test_ref, label_dict)

# Put the patch size according to you training here
patches_test = extract_patches_train(img_test_normalized, args.patch_size)
patches_test_ref = extract_patches_test(binary_img_test_ref, args.patch_size)

print(patches_test.shape)

# Load model
# another_strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
# with another_strategy.scope():
another_strategy = tf.distribute.MirroredStrategy()
with another_strategy.scope():
    model = load_model(args.model_path, compile=False)
model.summary()
# Prediction
patches_pred = Test(model, patches_test, args)
print('='*40)
print('[TEST]')

if args.use_multitasking:
    print(len(patches_pred))
    # print(patches_pred['seg'])
    print(type(patches_pred))
    print(model.output_names)
    preds = dict(zip(model.output_names, patches_pred))
    print(preds.keys())
    print(preds)
    # seg_preds = patches_pred[3]
    preds = patches_pred
    seg_preds = preds['seg']
    print(f'seg shape argmax: {seg_preds.shape}')
    seg_pred = np.argmax(seg_preds, axis=-1)
    print(f'seg shape argmax: {seg_pred.shape}')
    patches_pred = [preds['seg'], preds['bound'], preds['dist'], preds['color']]
else:
    seg_pred = patches_pred

# Metrics
true_labels = np.reshape(patches_test_ref, (patches_test_ref.shape[0] *
                                            patches_test_ref.shape[1] *
                                            patches_test_ref.shape[2]))

predicted_labels = np.reshape(seg_pred, (seg_pred.shape[0] *
                                         seg_pred.shape[1] *
                                         seg_pred.shape[2]))

# Metrics
metrics = compute_metrics(true_labels, predicted_labels)
confusion_matrix = confusion_matrix(true_labels, predicted_labels)

print('Confusion  matrix \n', confusion_matrix)
print()
print('Accuracy: ', metrics[0])
print('F1score: ', metrics[1])
print('Recall: ', metrics[2])
print('Precision: ', metrics[3])

# Reconstruct entire image segmentation predction
img_reconstructed = pred_recostruction(args.patch_size, seg_pred,
                                       binary_img_test_ref, img_type=1)
img_reconstructed_rgb = convert_preds2rgb(img_reconstructed,
                                          label_dict)

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

plt.imsave(os.path.join(args.output_path, 'pred_seg_reconstructed.jpeg'),
           img_reconstructed_rgb)

# Visualize inference per class
if args.use_multitasking:

    for i in range(len(patches_test)):
        print(f'Patch: {i}')
        # Plot predictions for each class and each task; Each row corresponds to a
        # class and has its predictions of each task
        fig1, axes = plt.subplots(nrows=args.num_classes, ncols=7, figsize=(15, 10))
        img = patches_test[i]
        img = (img * np.array([255, 255, 255])).astype(np.uint8)
        img_ref = patches_test_ref[i]
        img_ref_h = tf.keras.utils.to_categorical(img_ref, args.num_classes)
        bound_ref_h = get_boundary_label(img_ref_h)
        dist_ref_h = get_distance_label(img_ref_h)
        # Put the first plot as the patch to be observed on each row
        for n_class in range(args.num_classes):
            axes[n_class, 0].imshow(img)
            # Loop the columns to display each task prediction and reference
            # Remeber we are not displaying color preds here, since this task
            # do not use classes
            # Multiply by 2 cause its always pred and ref side by side
            for task in range(len(patches_pred) - 1):
                task_pred = patches_pred[task]
                col_ref = (task + 1)*2
                print(task_pred.shape)
                axes[n_class, col_ref].imshow(task_pred[i, :, :, n_class],
                                              cmap=cm.Greys_r)
                col = col_ref - 1
                if task == 0:
                    # Segmentation
                    axes[n_class, col].imshow(img_ref_h[:, :, n_class],
                                              cmap=cm.Greys_r)
                elif task == 1:
                    # Boundary
                    print(f' bound class: {n_class}')
                    axes[n_class, col].imshow(bound_ref_h[:, :, n_class],
                                              cmap=cm.Greys_r)
                elif task == 2:
                    # Distance Transform
                    axes[n_class, col].imshow(dist_ref_h[:, :, n_class],
                                              cmap=cm.Greys_r)
        axes[0, 0].set_title('Patch')
        axes[0, 1].set_title('Seg Ref')
        axes[0, 2].set_title('Seg Pred')
        axes[0, 3].set_title('Bound Ref')
        axes[0, 4].set_title('Bound Pred')
        axes[0, 5].set_title('Dist Ref')
        axes[0, 6].set_title('Dist Pred')

        for n_class in range(args.num_classes):
            axes[n_class, 0].set_ylabel(f'Class {n_class}')

        plt.savefig(os.path.join(args.output_path, f'pred{i}_classes.jpg'))

        # Color
        fig2, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
        ax1.set_title('Original')
        ax1.imshow(img)
        ax2.set_title('Pred HSV in RGB')
        task = 3
        hsv_pred = patches_pred[task][i]
        # print(f'HSV max {i}: {hsv_patch.max()}, HSV min: {hsv_patch.min()}')
        # As long as the normalization process was just img = img / 255
        hsv_patch = (hsv_pred * np.array([179, 255, 255])).astype(np.uint8)
        rgb_patch = cv2.cvtColor(hsv_patch, cv2.COLOR_HSV2RGB)
        ax2.imshow(rgb_patch)
        # ax3.set_title('Difference between both')
        # diff = np.mean(rgb_patch - img, axis=-1)
        # diff = 2*(diff-diff.min())/(diff.max()-diff.min()) - np.ones_like(diff)
        # # ax3.imshow(img - rgb_patch)
        # im = ax3.imshow(diff, cmap=cm.Greys_r)
        # colorbar(im, ax3, fig2)
        ax3.set_title('Difference between both')
        hsv_label = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        diff = np.mean(hsv_patch - hsv_label, axis=-1)
        diff = 2*(diff-diff.min())/(diff.max()-diff.min()) - np.ones_like(diff)
        im = ax3.imshow(diff, cmap=cm.Greys_r)
        colorbar(im, ax3, fig2)

        plt.savefig(os.path.join(args.output_path, f'pred{i}_color.jpg'))
        plt.show()
        plt.close()
