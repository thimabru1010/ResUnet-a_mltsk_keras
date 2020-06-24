import utils
import time
from utils import np, plt, load_tiff_image, load_SAR_image, compute_metrics, data_augmentation, unet, normalization, \
RGB_image, extract_patches, patch_tiles, bal_aug_patches, extrac_patch2, test_FCN, pred_recostruction, \
weighted_categorical_crossentropy, mask_no_considered, tf, Adam, prediction, load_model, confusion_matrix, \
EarlyStopping, ModelCheckpoint, identity_block, ResNet50, color_map, SGD, \
load_npy_image

from ResUnet_a.model import Resunet_a
from ResUnet_a.model2 import Resunet_a2, Resunet_a2_multitasking
from multitasking_utils import get_boundary_labels, get_distance_labels
import argparse
import os

from skimage.util.shape import view_as_windows
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--resunet_a",
    help="choose resunet-a model or not", type=int, default=0)
parser.add_argument("--multitasking",
    help="choose resunet-a model or not", type=int, default=0)
args = parser.parse_args()

def extract_patches_hw(image, reference, patch_size, stride):
    patches_out = []
    label_out = []
    window_shape = patch_size
    window_shape_array = (window_shape, window_shape, image.shape[2])
    window_shape_ref = (window_shape, window_shape)
    patches_array = np.array(view_as_windows(image, window_shape_array, step = stride))

    patches_ref = np.array(view_as_windows(reference, window_shape_ref, step = stride))

    print(patches_array.shape)
    num_row,num_col,p,row,col,depth = patches_array.shape

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

def Test(model, patch_test):
    result = model.predict(patch_test)
    predicted_class = np.argmax(result, axis=-1)
    return predicted_class

def compute_metrics_hw(true_labels, predicted_labels):
    accuracy = 100*accuracy_score(true_labels, predicted_labels)
    #avg_accuracy = 100*accuracy_score(true_labels, predicted_labels, average=None)
    f1score = 100*f1_score(true_labels, predicted_labels, average=None)
    recall = 100*recall_score(true_labels, predicted_labels, average=None)
    precision = 100*precision_score(true_labels, predicted_labels, average=None)
    return accuracy, f1score, recall, precision

root_path = './DATASETS/homework3_npy'
# Load images
img_train_path = 'Image_Train.npy'
img_train = load_npy_image(os.path.join(root_path, img_train_path)).astype(np.float32)
# Normalizes the image
img_train_normalized = normalization(img_train)
# Transform the image into W x H x C shape
img_train_normalized = img_train_normalized.transpose((1,2,0))
print(img_train_normalized.shape)

# Load reference
img_train_ref_path = 'Reference_Train.npy'
img_train_ref = load_npy_image(os.path.join(root_path, img_train_ref_path))
img_train_ref = img_train_ref.transpose((1,2,0))
print(img_train_ref.shape)


# Create binarized matrix
w = img_train_ref.shape[0]
h = img_train_ref.shape[1]
c = img_train_ref.shape[2]
#binary_img_train_ref = np.zeros((1,w,h))
binary_img_train_ref = np.full((w,h), -1)
label_dict = {'(255, 255, 255)': 0, '(0, 255, 0)': 1, '(0, 255, 255)': 2, '(0, 0, 255)': 3, '(255, 255, 0)': 4}
for i in range(w):
    for j in range(h):
        r = img_train_ref[i][j][0]
        g = img_train_ref[i][j][1]
        b = img_train_ref[i][j][2]
        rgb = (r,g,b)
        rgb_key = str(rgb)
        binary_img_train_ref[i][j] = label_dict[rgb_key]

number_class = 5
patch_size = 256
stride = patch_size


#stride = patch_size
patches_tr, patches_tr_ref = extract_patches_hw(img_train_normalized, binary_img_train_ref, patch_size, stride)

# Load images

# patches_tr_aug, patches_tr_ref_aug = bal_aug_patches(percent, patch_size, patches_tr, patches_tr_ref)
# patches_tr_ref_aug_h = tf.keras.utils.to_categorical(patches_tr_ref_aug, number_class)
# Creates one-hot encoding for segmentation
patches_tr_ref_h = tf.keras.utils.to_categorical(patches_tr_ref, number_class)

print('[DEBUG LABELS]')
# Create labels for boundary
patches_bound_labels = get_boundary_labels(patches_tr)

# Create labels for distance
patches_dist_labels = get_distance_labels(patches_tr)

patches_tr , patches_tr_ref_h = shuffle(patches_tr , patches_tr_ref_h , random_state = 42)

patches_tr, patches_val, patches_tr_ref_h, patches_val_ref_h = train_test_split(patches_tr, patches_tr_ref_h, test_size=0.2, random_state=42)
print(patches_tr.shape, patches_val.shape)
print(patches_tr_ref_h.shape, patches_val_ref_h.shape)


# Validation tiles

# patches_val_aug, patches_val_ref_aug = bal_aug_patches(percent, patch_size, patches_val, patches_val_ref)
#
# patches_val_ref_aug_h = tf.keras.utils.to_categorical(patches_val_ref_aug, number_class)

#%%
start_time = time.time()
exp = 1
rows = patch_size
cols = patch_size
channels = 3
adam = Adam(lr = 0.01 , beta_1=0.9)
sgd = SGD(lr=0.01,momentum=0.8)
batch_size = 1

#weights = [0.5, 0.5, 0]
#weights = [weight0, weight1, 0]
weights = [  4.34558461   ,2.97682037   ,3.92124661   ,5.67350328 ,374.0300152 ]
print('='*60)
print(weights)
loss = weighted_categorical_crossentropy(weights)
if args.multitasking:
    loss = multitasking_weighted_categorical_crossentropy(weights)

if args.resunet_a == True:

    if args.multitasking:
        print('Multitasking enabled!')
        resuneta = Resunet_a2_multitasking((rows, cols, channels))
        model = resuneta.model
        model.summary()
        losses = {
        	"segmentation": "categorical_crossentropy",
        	"boundary": "categorical_crossentropy",
            "distance": "categorical_crossentropy",
            "color": "categorical_crossentropy",
        }
        lossWeights = {"segmentation": 1.0, "boundary": 1.0, "distance": 1.0,
        "color": 1.0}
        model.compile(optimizer=adam, loss=losses, loss_weights=lossWeights, metrics=['accuracy'])
    else:
        resuneta = Resunet_a2((rows, cols, channels))
        model = resuneta.model
        model.summary()
        model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])

    #model = Resunet_a((channels, cols, rows))
    print('ResUnet-a compiled!')
else:
    model = unet((rows, cols, channels))
    model.summary()

    model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
#model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
# print model information


filepath = './models/'
# define early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath+'unet_exp_'+str(exp)+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [earlystop, checkpoint]

# train the model
start_training = time.time()
model_info = model.fit(patches_tr, patches_tr_ref_h, batch_size=batch_size, epochs=100, callbacks=callbacks_list, verbose=2, validation_data= (patches_val, patches_val_ref_h) )
end_training = time.time() - start_time

#%% Test model
# Creation of mask with test tiles

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
patches_test = extract_patches_train(img_test_normalized, patch_size)
patches_test_ref = extract_patches_test(binary_img_test_ref, patch_size)

#% Load model
model = load_model(filepath+'unet_exp_'+str(exp)+'.h5', compile=False)
area = 11
# Prediction
# Test the model
patches_pred = Test(model, patches_test)
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

plt.imsave('img_reconstructed_rgb.jpeg', img_reconstructed_rgb)
