import utils
import time
from utils import np, plt, load_tiff_image, load_SAR_image, compute_metrics, data_augmentation, unet, normalization, \
RGB_image, extract_patches, patch_tiles, bal_aug_patches, extrac_patch2, test_FCN, pred_recostruction, \
weighted_categorical_crossentropy, mask_no_considered, tf, Adam, prediction, load_model, confusion_matrix, \
EarlyStopping, ModelCheckpoint, identity_block, ResNet50, color_map, SGD, \
load_npy_image

from ResUnet_a.model import Resunet_a
from ResUnet_a.model2 import Resunet_a2
from multitasking_utils import get_boundary_labels, get_distance_labels, get_color_labels
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

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

parser = argparse.ArgumentParser()
parser.add_argument("--multitasking",
    help="choose resunet-a model or not", type=int, default=0)
args = parser.parse_args()

def Test(model, patch_test, args):
    result = model.predict(patch_test)
    print(len(result))
    if args.multitasking:
        print('Multitasking Enabled!')
        print(result[0].shape)
        #predicted_class = np.argmax(result[2], axis=-1)
        predicted_class = result
        #print(predicted_class.shape)
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

def pred_recostruction(patch_size, pred_labels, binary_img_test_ref, img_type=1):
    # Patches Reconstruction
    if img_type == 1:
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

#%% Test model
# Creation of mask with test tiles
root_path = './DATASETS/homework3_npy'

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
#binary_img_train_ref = np.zeros((1,w,h))
binary_img_test_ref = np.full((w,h), -1)
# Dictionary used in training
label_dict = {'(255, 255, 255)': 0, '(0, 255, 0)': 1, '(0, 255, 255)': 2, '(0, 0, 255)': 3, '(255, 255, 0)': 4}
label = 0
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
patch_size = 256
patches_test = extract_patches_train(img_test_normalized, patch_size)
patches_test_ref = extract_patches_test(binary_img_test_ref, patch_size)

#% Load model
filepath = './models/'
exp=4
model = load_model(filepath+'unet_exp_'+str(exp)+'.h5', compile=False)
model.summary()
# Prediction
# Test the model
patches_pred = Test(model, patches_test, args)
# result = model.predict(patches_test)
# patches_pred = np.argmax(result[0], axis=-1)
print('='*40)
#print(len(result))
print('[TEST]')
print()
#print(patches_pred.shape)

# Metrics
# true_labels = np.reshape(patches_test_ref, (patches_test_ref.shape[0]* patches_test_ref.shape[1]*patches_test_ref.shape[2]))
# # true_labels = np.reshape(patches_test_ref, (patches_test_ref.shape[0]* patches_test_ref.shape[1]*patches_test_ref.shape[2]))
#
# predicted_labels = np.reshape(patches_pred, (patches_pred.shape[0]* patches_pred.shape[1]*patches_pred.shape[2]))
#
# # Metrics
# metrics = compute_metrics(true_labels,predicted_labels)
# cm = confusion_matrix(true_labels, predicted_labels, labels=[0,1,2,3,4])
#
# print('Confusion  matrix \n', cm)
# print()
# print('Accuracy: ', metrics[0])
# print('F1score: ', metrics[1])
# print('Recall: ', metrics[2])
# print('Precision: ', metrics[3])

print(patches_test_ref.shape)
#patches_test_ref_h = tf.keras.utils.to_categorical(patches_test_ref, 5)
#patches_test_ref_h = tf.keras.utils.to_categorical(patches_pred, 5)
# patches_pred = np.sum(get_pred_distance_labels(patches_test_ref_h), axis=-1)/5
# print(patches_pred.shape)
#bounds = get_boundary_labels(patches_test_ref_h)
# for i in range(len(bounds)):
#     print(bounds[i])
# patches_pred = np.sum(bounds, axis=-1)/5
# print(patches_pred.shape)
# print(patches_test.shape)
# patches_pred = get_color_labels(patches_test.astype(np.uint8))
# print(patches_pred.shape)
#patches_pred = get_pred_distance_labels(patches_test_ref_h)[:,:,:,0]
#patches_pred = np.max(patches_pred, axis=-1)
# for i in range(len(patches_pred)):
#     print(patches_pred[i])
# patches_pred = cv2.normalize(patches_pred, patches_pred, 0, 1.0, cv2.NORM_MINMAX)
if not args.multitasking:
    img_reconstructed = pred_recostruction(patch_size, patches_pred, binary_img_test_ref, 1)
    img_reconstructed_rgb = reconstruction_rgb_prdiction_patches(img_reconstructed, label_dict)

    plt.imsave(f'img_reconstructed_rgb_exp{exp}.jpeg', img_reconstructed_rgb)
else:
    # Boundaries
    print(patches_pred[1].shape)
    pred_bound = np.sum(patches_pred[1],axis=-1)
    pred_bound /= pred_bound.max()
    print(pred_bound.shape)
    img_reconstructed = pred_recostruction(patch_size, pred_bound, binary_img_test_ref, 1)

    fig1, ax1 = plt.subplots()
    im = ax1.imshow(img_reconstructed,cmap=cm.Greys_r)
    fig1.colorbar(im, ax=ax1)
    plt.savefig('teste1.jpg')
    plt.show()

    #plt.imsave(f'img_reconstructed_pred_bound_exp{exp}.jpeg', img_reconstructed)

    # Distance transform
    pred_dist = np.sum(np.clip(patches_pred[2],a_min=0.3,a_max=1.0),axis=-1)

    pred_dist = (pred_dist - pred_dist.min())/(pred_dist.max() - pred_dist.min())
    img_reconstructed = pred_recostruction(patch_size, pred_dist, binary_img_test_ref, 1)

    fig2, ax2 = plt.subplots()
    im = ax2.imshow(img_reconstructed,cmap=cm.Greys_r)
    fig2.colorbar(im, ax=ax2)
    plt.savefig('teste2.jpg')
    plt.show()

    # Color transform
    hsv_patches_label = get_color_labels(patches_test.astype(np.uint8))
    fig3, (ax3, ax4, ax5, ax6) = plt.subplots(1, 4)
    rgb_img = hsv_to_rgb(patches_pred[3])
    img_reconstructed_rgb = pred_recostruction(patch_size, rgb_img, binary_img_test_ref, 2)
    ax3.imshow(img_reconstructed_rgb.astype(np.uint8),rasterized=True)
    diff = np.mean(patches_pred[3] - hsv_patches_label ,axis=-1)
    diff =  2*(diff-diff.min())/(diff.max()-diff.min()) - np.ones_like(diff)
    #
    # fig2, ax2 = plt.subplots()
    # im = ax2.imshow(img_reconstructed,cmap=cm.Greys_r)
    # fig2.colorbar(im, ax=ax2)
    # plt.savefig('teste3.jpg')

    img_reconstructed = pred_recostruction(patch_size, diff, binary_img_test_ref, 1)
    im = ax4.imshow(img_reconstructed,cmap=cm.Greys_r)
    fig3.colorbar(im, ax=ax4)

    ax5.imshow(img_test.transpose((1,2,0)).astype(np.uint8))

    img_reconstructed_rgb = pred_recostruction(patch_size, hsv_to_rgb(hsv_patches_label), binary_img_test_ref, 2)
    ax6.imshow(img_reconstructed_rgb)

    plt.show()

    plt.imshow(img_reconstructed_rgb)
    plt.show()

    #plt.imsave(f'img_reconstructed_pred_dist_exp{exp}.jpeg', img_reconstructed)
