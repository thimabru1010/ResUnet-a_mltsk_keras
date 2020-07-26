import utils
import time
from utils import np, plt, load_tiff_image, load_SAR_image, compute_metrics, data_augmentation, unet, normalization, \
RGB_image, extract_patches, patch_tiles, bal_aug_patches, extrac_patch2, test_FCN, pred_recostruction, \
weighted_categorical_crossentropy, mask_no_considered, tf, Adam, prediction, load_model, confusion_matrix, \
EarlyStopping, ModelCheckpoint, identity_block, ResNet50, color_map, SGD, \
load_npy_image

from ResUnet_a.model import Resunet_a
from ResUnet_a.model2 import Resunet_a2
from multitasking_utils import get_boundary_labels, get_distance_labels, get_color_labels, Tanimoto_dual_loss
import argparse
import os

from skimage.util.shape import view_as_windows
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import gc
import psutil

from CustomDataGenerator import Mygenerator, Mygenerator_multitasking
import ast

from multitasking_utils import get_boundary_labels, get_distance_labels, get_color_labels

# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu_devices[0], True)

parser = argparse.ArgumentParser()
parser.add_argument("--resunet_a",
    help="choose resunet-a model or not", type=int, default=0)
parser.add_argument("--multitasking",
    help="choose resunet-a model or not", type=int, default=0)
parser.add_argument("--gpu_parallel",
    help="choose 1 to train one multiple gpu", type=int, default=0)
args = parser.parse_args()

def extract_patches_hw(image, reference, patch_size, stride):
    patches_out = []
    label_out = []
    window_shape = patch_size
    window_shape_array = (window_shape, window_shape, image.shape[2])
    window_shape_ref = (window_shape, window_shape)
    patches_array = np.array(view_as_windows(image, window_shape_array, step = stride))

    patches_ref = np.array(view_as_windows(reference, window_shape_ref, step = stride))

    print('Patches extraidos')
    print(patches_array.shape)
    num_row,num_col,p,row,col,depth = patches_array.shape

    print('fazendo reshape')
    patches_array = patches_array.reshape(num_row*num_col,row,col,depth)
    print(patches_array.shape)
    patches_ref = patches_ref.reshape(num_row*num_col,row,col)
    print(patches_ref.shape)

    return patches_array, patches_ref

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

def binarize_matrix(img_train_ref, label_dict):
    # Create binarized matrix
    w = img_train_ref.shape[0]
    h = img_train_ref.shape[1]
    c = img_train_ref.shape[2]
    #binary_img_train_ref = np.zeros((1,w,h))
    binary_img_train_ref = np.full((w,h), -1)
    for i in range(w):
        for j in range(h):
            r = img_train_ref[i][j][0]
            g = img_train_ref[i][j][1]
            b = img_train_ref[i][j][2]
            rgb = (r,g,b)
            rgb_key = str(rgb)
            binary_img_train_ref[i][j] = label_dict[rgb_key]

    return binary_img_train_ref

def bal_aug_patches2(percent, patch_size, patches_img, patches_ref):
    patches_images = []
    patches_labels = []

    for i in range(0,len(patches_img)):

        patch_img = patches_img[i]
        patch_label = patches_ref[i]
        img_aug, label_aug = data_augmentation(patch_img, patch_label)
        patches_images.append(img_aug)
        patches_labels.append(label_aug)

    patches_bal = np.concatenate(patches_images).astype(np.float32)
    labels_bal = np.concatenate(patches_labels).astype(np.float32)
    return patches_bal, labels_bal

if args.gpu_parallel:
    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')
else:
    strategy = None

root_path = './DATASETS/homework3_npy'
# Load images
img_train_path = 'Image_Train.npy'
img_train = load_npy_image(os.path.join(root_path, img_train_path)).astype(np.float32)
# Normalizes the image
img_train_normalized = normalization(img_train)
# Transform the image into W x H x C shape
img_train_normalized = img_train_normalized.transpose((1,2,0))
print('Imagem RGB')
print(img_train_normalized.shape)

# Load reference
img_train_ref_path = 'Reference_Train.npy'
img_train_ref = load_npy_image(os.path.join(root_path, img_train_ref_path))
img_train_ref = img_train_ref.transpose((1,2,0))
print('Imagem de referencia')
print(img_train_ref.shape)

label_dict = {'(255, 255, 255)': 0, '(0, 255, 0)': 1, '(0, 255, 255)': 2, '(0, 0, 255)': 3, '(255, 255, 0)': 4}

binary_img_train_ref = binarize_matrix(img_train_ref, label_dict)
del img_train_ref

number_class = 5
patch_size = 256
stride = patch_size // 8


#stride = patch_size
patches_tr, patches_tr_ref = extract_patches_hw(img_train_normalized, binary_img_train_ref, patch_size, stride)
print('patches extraidos!')
# patches_tr, patches_tr_ref = bal_aug_patches2(percent, patch_size, patches_tr, patches_tr_ref)
process = psutil.Process(os.getpid())
print('[CHECKING MEMORY]')
#print(process.memory_info().rss)
print(process.memory_percent())
del binary_img_train_ref, img_train_normalized, img_train
#print(process.memory_info().rss)
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
        np.save(os.path.join(folder_path, 'train', filename(i*5 + j)), img_aug[j])
        np.save(os.path.join(folder_path, 'labels/seg', filename(i*5 + j)), label_aug_h[j])
        np.save(os.path.join(folder_path, 'labels/bound', filename(i*5 + j)), patches_bound_labels_h[j])
        np.save(os.path.join(folder_path, 'labels/dist', filename(i*5 + j)), patches_dist_labels_h[j])
        np.save(os.path.join(folder_path, 'labels/color', filename(i*5 + j)), patches_color_labels_h[j])
