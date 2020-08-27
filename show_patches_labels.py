# from utils import np, plt, load_tiff_image, load_SAR_image, compute_metrics, data_augmentation, unet, normalization, \
# RGB_image, extract_patches, patch_tiles, bal_aug_patches, extrac_patch2, test_FCN, pred_recostruction, \
# weighted_categorical_crossentropy, mask_no_considered, tf, Adam, prediction, load_model, confusion_matrix, \
# EarlyStopping, ModelCheckpoint, identity_block, ResNet50, color_map, SGD, \
# load_npy_image

from utils import np, load_npy_image, normalization, data_augmentation, plt
import tensorflow as tf

from multitasking_utils import get_boundary_labels, get_distance_labels, \
    get_color_labels
import argparse
import os

from skimage.util.shape import view_as_windows

import gc
import psutil

import cv2
from osgeo import ogr, gdal
from matplotlib import cm

parser = argparse.ArgumentParser()
parser.add_argument("--multitasking",
                    help="choose resunet-a model or not", type=int, default=0)
args = parser.parse_args()


# Functions
def load_tiff_image(patch):
    # Read tiff Image
    print(patch)
    gdal_header = gdal.Open(patch)
    img = gdal_header.ReadAsArray()
    return img


def extract_patches_hw(image, reference, patch_size, stride):
    window_shape = patch_size
    window_shape_array = (window_shape, window_shape, image.shape[2])
    window_shape_ref = (window_shape, window_shape)
    patches_array = np.array(view_as_windows(image,
                                             window_shape_array, step=stride))

    patches_ref = np.array(view_as_windows(reference,
                                           window_shape_ref, step=stride))

    print('Patches extraidos')
    print(patches_array.shape)
    num_row, num_col, p, row, col, depth = patches_array.shape

    print('fazendo reshape')
    patches_array = patches_array.reshape(num_row*num_col, row, col, depth)
    print(patches_array.shape)
    patches_ref = patches_ref.reshape(num_row*num_col, row, col)
    print(patches_ref.shape)

    return patches_array, patches_ref


def binarize_matrix(img_train_ref, label_dict):
    # Create binarized matrix
    w = img_train_ref.shape[0]
    h = img_train_ref.shape[1]
    # c = img_train_ref.shape[2]
    # binary_img_train_ref = np.zeros((1,w,h))
    binary_img_train_ref = np.full((w, h), -1)
    for i in range(w):
        for j in range(h):
            r = img_train_ref[i][j][0]
            g = img_train_ref[i][j][1]
            b = img_train_ref[i][j][2]
            rgb = (r, g, b)
            rgb_key = str(rgb)
            binary_img_train_ref[i][j] = label_dict[rgb_key]

    return binary_img_train_ref


root_path = './DATASETS/homework3'
# Load images
img_train_path = 'Image_Train.tif'
img_train = load_tiff_image(os.path.join(root_path,
                                        img_train_path))
# img_train = plt.imread(os.path.join(root_path,
#                                    img_train_path))
# print(type(img_train))
# Normalizes the image
# img_train_normalized = normalization(img_train)*255
# Transform the image into W x H x C shape
# img_train_normalized = img_train_normalized.transpose((1, 2, 0))
print('Imagem RGB')
# print(img_train)
print(img_train.shape)
img_train = img_train.transpose((1, 2, 0))
print(img_train.shape)
# print(img_train)
from skimage.transform import resize
# img_train = cv2.resize(img_train, (500, 500), cv2.INTER_AREA)
# img_train = resize(img_train, (500, 500))
# #cv2.imshow('teste', img_train)
# plt.imshow(img_train)
# plt.show()
# print('image showed')
#cv2.waitKey(0)
# print(img_train_normalized.shape)

# Load reference
img_train_ref_path = 'Reference_Train.tif'
# img_train_ref_path = 'Image_Train_ref.jpeg'
print(os.path.join(root_path, img_train_ref_path))
img_train_ref = load_tiff_image(os.path.join(root_path, img_train_ref_path))
img_train_ref = img_train_ref.transpose((1, 2, 0))
# img_train_ref = plt.imread(os.path.join(root_path, img_train_ref_path))
print('Imagem de referencia')
print(img_train_ref.shape)

# img_train_ref = resize(img_train_ref, (500, 500))
# #cv2.imshow('teste', img_train)
# img_train_ref = img_train_ref[::-1]
# plt.imshow(img_train_ref)
# plt.show()

label_dict = {'(255, 255, 255)': 0, '(0, 255, 0)': 1,
              '(0, 255, 255)': 2, '(0, 0, 255)': 3, '(255, 255, 0)': 4}

binary_img_train_ref = binarize_matrix(img_train_ref, label_dict)
del img_train_ref

number_class = 5
patch_size = 256
stride = patch_size // 1


# stride = patch_size
patches_tr, patches_tr_ref = extract_patches_hw(img_train,
                                                binary_img_train_ref,
                                                patch_size, stride)
print('patches extraidos!')
process = psutil.Process(os.getpid())
print('[CHECKING MEMORY]')
# print(process.memory_info().rss)
print(process.memory_percent())
del binary_img_train_ref, img_train
# print(process.memory_info().rss)
print(process.memory_percent())
gc.collect()
print('[GC COLLECT]')
print(process.memory_percent())


def filename(i):
    return f'patch_{i}.npy'


def get_boundary_label(label, _kernel_size = (3,3)):

    label = label.copy()
    h, w, c = label.shape
    # bounds = np.empty_like(labels,dtype=np.float32)
    # bounds = np.empty_like(label,dtype=np.int8)
    for channel in range(c):
        #print(label)
        label = label.astype(np.uint8)
        #print(label)
        #temp = cv2.Canny(label[:,:,channel],0,1)
        bound = cv2.Canny(label[:, :, channel], 0,1)
        label[:, :, channel] = cv2.dilate(bound, cv2.getStructuringElement(cv2.MORPH_CROSS,_kernel_size) ,iterations = 1)

        # bound = bound.astype(np.float32)
        # bounds /= 255.

        # bounds[n,:,:,channel] = bound

    # bounds = bounds.astype(np.float32)
    # bounds /= 255.
    return label

def get_distance_label(label):
    label = label.copy()
    #print (label.shape)
    dists = np.empty_like(label,dtype=np.float32)
    print('[DEBUG]')
    for channel in range(label.shape[2]):
        patch = label[:, :, channel].astype(np.uint8)
        #print(patch.shape)
        # print(cv2.distanceTransform(label[:, :, channel], cv2.DIST_L2, 0).shape)
        dist = cv2.distanceTransform(patch, cv2.DIST_L2, 0)
        dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        dists[:, :, channel] = dist

    return dists

def show_each_channel(img, axes, row):
    w, h, channel = img.shape
    for c in range(channel):
        axes[row, c].imshow(img[:, :, c], cmap=cm.Greys_r)
    #     plt.show()
    # plt.close()

print(f'Number of patches: {len(patches_tr)}')
print(f'Number of patches expected: {len(patches_tr)*5}')
for i in range(len(patches_tr)):
    # (axis_seg, axis_bound, axis_dist, axis_color)
    fig1, axes = plt.subplots(nrows=4, ncols=5, figsize=(12, 9))
    img_aug = patches_tr[i]
    label_aug = patches_tr_ref[i]
    print('seg')
    axes[0, 0].set_ylabel('Segmentation')
    label_aug_h = tf.keras.utils.to_categorical(label_aug, number_class)
    print(label_aug_h.shape)
    show_each_channel(label_aug_h, axes, row=0)
    # All multitasking labels are saved in one-hot
    # Create labels for boundary
    print('bound')
    axes[1, 0].set_ylabel('Boundary')
    patches_bound_labels_h = get_boundary_label(label_aug_h)
    show_each_channel(patches_bound_labels_h, axes, row=1)
    # Create labels for distance
    print('dist')
    axes[2, 0].set_ylabel('Distance')
    patches_dist_labels_h = get_distance_label(label_aug_h)
    show_each_channel(patches_dist_labels_h, axes, row=2)
    # Create labels for color
    print('color')
    axes[3, 0].set_ylabel('Boundary')
    hsv_patch = cv2.cvtColor(patches_tr[i], cv2.COLOR_RGB2HSV)
    axes[3, 0].imshow(hsv_patch)
    axes[3, 1].imshow(patches_tr[i])
    axes[3, 2].imshow(patches_tr[i])
    axes[3, 3].imshow(patches_tr[i])
    axes[3, 4].imshow(patches_tr[i])
    plt.show()
    # patches_color_labels_h = get_color_labels(patches_tr)
