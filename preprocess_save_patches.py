# from utils import np, plt, load_tiff_image, load_SAR_image, compute_metrics, data_augmentation, unet, normalization, \
# RGB_image, extract_patches, patch_tiles, bal_aug_patches, extrac_patch2, test_FCN, pred_recostruction, \
# weighted_categorical_crossentropy, mask_no_considered, tf, Adam, prediction, load_model, confusion_matrix, \
# EarlyStopping, ModelCheckpoint, identity_block, ResNet50, color_map, SGD, \
# load_npy_image

from utils import np, load_npy_image, normalization, data_augmentation
import tensorflow as tf

from multitasking_utils import get_boundary_labels, get_distance_labels, \
    get_color_labels
import argparse
import os

from skimage.util.shape import view_as_windows

import gc
import psutil


parser = argparse.ArgumentParser()
parser.add_argument("--multitasking",
                    help="choose resunet-a model or not", type=int, default=0)
args = parser.parse_args()


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


root_path = './DATASETS/homework3_npy'
# Load images
img_train_path = 'Image_Train.npy'
img_train = load_npy_image(os.path.join(root_path,
                                        img_train_path)).astype(np.float32)
# Normalizes the image
img_train_normalized = normalization(img_train)
# Transform the image into W x H x C shape
img_train_normalized = img_train_normalized.transpose((1, 2, 0))
print('Imagem RGB')
print(img_train_normalized.shape)

# Load reference
img_train_ref_path = 'Reference_Train.npy'
img_train_ref = load_npy_image(os.path.join(root_path, img_train_ref_path))
img_train_ref = img_train_ref.transpose((1, 2, 0))
print('Imagem de referencia')
print(img_train_ref.shape)

label_dict = {'(255, 255, 255)': 0, '(0, 255, 0)': 1,
              '(0, 255, 255)': 2, '(0, 0, 255)': 3, '(255, 255, 0)': 4}

binary_img_train_ref = binarize_matrix(img_train_ref, label_dict)
del img_train_ref

number_class = 5
patch_size = 128
stride = patch_size // 8


# stride = patch_size
patches_tr, patches_tr_ref = extract_patches_hw(img_train_normalized,
                                                binary_img_train_ref,
                                                patch_size, stride)
print('patches extraidos!')
process = psutil.Process(os.getpid())
print('[CHECKING MEMORY]')
# print(process.memory_info().rss)
print(process.memory_percent())
del binary_img_train_ref, img_train_normalized, img_train
# print(process.memory_info().rss)
print(process.memory_percent())
gc.collect()
print('[GC COLLECT]')
print(process.memory_percent())

print('saving images...')
folder_path = f'./DATASETS/patches_ps={patch_size}_stride={stride}'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    os.makedirs(os.path.join(folder_path, 'train'))
    os.makedirs(os.path.join(folder_path, 'labels'))
    os.makedirs(os.path.join(folder_path, 'labels/seg'))
    os.makedirs(os.path.join(folder_path, 'labels/bound'))
    os.makedirs(os.path.join(folder_path, 'labels/dist'))
    os.makedirs(os.path.join(folder_path, 'labels/color'))


def filename(i):
    return f'patch_{i}.npy'


print(f'Number of patches: {len(patches_tr)}')
print(f'Number of patches expected: {len(patches_tr)*5}')
for i in range(len(patches_tr)):
    img_aug, label_aug = data_augmentation(patches_tr[i], patches_tr_ref[i])
    label_aug_h = tf.keras.utils.to_categorical(label_aug, number_class)
    # All multitasking labels are saved in one-hot
    # Create labels for boundary
    patches_bound_labels_h = get_boundary_labels(label_aug_h)
    # Create labels for distance
    patches_dist_labels_h = get_distance_labels(label_aug_h)
    # Create labels for color
    patches_color_labels_h = get_color_labels(patches_tr)
    for j in range(len(img_aug)):
        np.save(os.path.join(folder_path, 'train', filename(i*5 + j)),
                img_aug[j])
        np.save(os.path.join(folder_path, 'labels/seg', filename(i*5 + j)),
                label_aug_h[j])
        np.save(os.path.join(folder_path, 'labels/bound', filename(i*5 + j)),
                patches_bound_labels_h[j])
        np.save(os.path.join(folder_path, 'labels/dist', filename(i*5 + j)),
                patches_dist_labels_h[j])
        np.save(os.path.join(folder_path, 'labels/color', filename(i*5 + j)),
                patches_color_labels_h[j])
