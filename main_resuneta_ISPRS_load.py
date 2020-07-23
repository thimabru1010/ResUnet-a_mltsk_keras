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

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

parser = argparse.ArgumentParser()
parser.add_argument("--resunet_a",
    help="choose resunet-a model or not", type=int, default=0)
parser.add_argument("--multitasking",
    help="choose resunet-a model or not", type=int, default=0)
parser.add_argument("--gpu_parallel",
    help="choose 1 to train one multiple gpu", type=int, default=0)
args = parser.parse_args()

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

def Train_model(net, patches_train, patches_tr_lb_h, patches_val, patches_val_lb_h, batch_size, epochs, patience, delta, x_shape_batch, y_shape_batch, seed):
    print('Start training...')
    print('='*60)
    print(f'Training on {len(patches_train)} images')
    print(f'Validating on {len(patches_val)} images')
    print('='*60)
    print(f'Epochs: {epochs}')
    best_score = 0
    min_loss = 200
    cont = 0
    total_train_loss = []
    total_train_acc = []
    total_val_loss = []
    total_val_acc =[]
    x_train_b = np.zeros(x_shape_batch)
    y_train_h_b = np.zeros(y_shape_batch)
    x_val_b = np.zeros(x_shape_batch)
    y_val_h_b = np.zeros(y_shape_batch)
    for epoch in range(epochs):
        loss_tr = np.zeros((1 , 2))
        loss_val = np.zeros((1 , 2))
        # Computing the number of batchs on training
        #n_batchs_tr = patches_train.shape[0]//batch_size
        n_batchs_tr = len(patches_train)//batch_size
        # Random shuffle the data
        patches_train , patches_tr_lb_h = shuffle(patches_train , patches_tr_lb_h , random_state = seed)

        # Training the network per batch
        for  batch in range(n_batchs_tr):
            # x_train_b = patches_train[batch * batch_size : (batch + 1) * batch_size , : , : , :]
            x_train_paths = patches_train[batch * batch_size : (batch + 1) * batch_size]
            # y_train_h_b = patches_tr_lb_h[batch * batch_size : (batch + 1) * batch_size , :, :, :]
            y_train_paths = patches_tr_lb_h[batch * batch_size : (batch + 1) * batch_size]
            for b in range(batch_size):
                x_train_b[b] = np.load(x_train_paths[b])
                y_train_h_b[b] = np.load(y_train_paths[b])

            loss_tr = loss_tr + net.train_on_batch(x_train_b , y_train_h_b)

        # Training loss
        loss_tr = loss_tr/n_batchs_tr
        #print("%d [Training loss: %f , Train acc.: %.2f%%]" %(epoch , loss_tr[0 , 0], 100*loss_tr[0 , 1]))

        # Computing the number of batchs on validation
        #n_batchs_val = patches_val.shape[0]//batch_size
        n_batchs_val = len(patches_val)//batch_size

        '''
            Talvez fosse bom deletar as matrizes :
            x_train_b
            y_train_h_b
            Antes de começar o validation
        '''

        # Evaluating the model in the validation set
        for  batch in range(n_batchs_val):
            # x_val_b = patches_val[batch * batch_size : (batch + 1) * batch_size , : , : , :]
            x_val_paths = patches_val[batch * batch_size : (batch + 1) * batch_size]
            y_val_paths = patches_val_lb_h[batch * batch_size : (batch + 1) * batch_size]

            for b in range(batch_size):
                x_val_b[b] = np.load(x_val_paths[b])
                y_val_h_b[b] = np.load(y_val_paths[b])

            loss_val = loss_val + net.test_on_batch(x_val_b , y_val_h_b)

        # validation loss
        loss_val = loss_val/n_batchs_val
        #print("%d [Validation loss: %f , Validation acc.: %.2f%%]" %(epoch , loss_val[0 , 0], 100*loss_val[0 , 1]))
        train_loss = loss_tr[0 , 0]
        train_acc = loss_tr[0 , 1]
        val_loss = loss_val[0 , 0]
        val_acc = loss_val[0 , 1]
        total_train_loss.append(train_loss)
        total_train_acc.append(train_acc)
        total_val_loss.append(val_loss)
        total_val_acc.append(val_acc)
        print(f"Epoch: {epoch} \t Training loss: {train_loss :.2f} \t Train acc.: {100*train_acc:.2f}% \t Validation loss: {val_loss :.2f} \t Validation acc.: {100*val_acc:.2f}%")
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
                return total_train_loss, total_train_acc, total_val_loss, total_val_acc
        else:
            cont = 0
            #best_score = score
            min_loss = val_loss
            print("Saving best model...")
            net.save('weights/best_model.h5')

    return total_train_loss, total_train_acc, total_val_loss, total_val_acc

if args.gpu_parallel:
    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')
else:
    strategy = None

root_path = './DATASETS/patches_ps=256_stride=32'
train_path = os.path.join(root_path, 'train')
ref_path = os.path.join(root_path, 'labels')
patches_train = [os.path.join(train_path,name) for name in os.listdir(train_path)]
patches_tr_lb_h = [os.path.join(ref_path, name) for name in os.listdir(ref_path)]

patches_train, patches_val, patches_tr_lb_h, patches_val_lb_h = train_test_split(patches_train, patches_tr_lb_h, test_size=0.2, random_state=42)

number_class = 5
patch_size = 256
stride = patch_size // 8
batch_size = 1
epochs = 100
seed = 42


if args.multitasking:
    print('[DEBUG LABELS]')
    # Create labels for boundary
    patches_bound_labels = get_boundary_labels(patches_tr_ref_h)
    print(patches_bound_labels.shape)

    # Create labels for distance
    patches_dist_labels = get_distance_labels(patches_tr_ref_h)
    print(patches_dist_labels.shape)

    # Create labels for color
    patches_color_labels = get_color_labels(patches_tr)
    print(patches_color_labels.shape)

    print(patches_tr.shape)

    print(patches_tr_ref_h.shape)

    # patches_tr , patches_tr_ref_h = shuffle(patches_tr , patches_tr_ref_h , random_state = 42)

    patches_tr, patches_val, patches_tr_ref_h, patches_val_ref_h, patches_bound_labels_tr, patches_bound_labels_val, patches_dist_labels_tr, patches_dist_labels_val, patches_color_labels_tr, patches_color_labels_val   = train_test_split(patches_tr, patches_tr_ref_h, patches_bound_labels, patches_dist_labels, patches_color_labels,  test_size=0.2, random_state=42)

    y_fit={"segmentation": patches_tr_ref_h, "boundary": patches_bound_labels_tr, "distance":  patches_dist_labels_tr, "color": patches_color_labels_tr}

    val_fit={"segmentation": patches_val_ref_h, "boundary": patches_bound_labels_val, "distance":  patches_dist_labels_val, "color": patches_color_labels_val}

    # Create generator
    batch_size = 1
    train_generator = Mygenerator_multitasking(patches_tr, y_fit, batch_size=batch_size)
    val_generator = Mygenerator_multitasking(patches_val, val_fit, batch_size=batch_size)

    # train_generator = Mygenerator(patches_tr, patches_tr_ref_h, batch_size=batch_size)
    # val_generator = Mygenerator(patches_val, patches_val_ref_h, batch_size=batch_size)

#%%
start_time = time.time()
exp = 1
rows = patch_size
cols = patch_size
channels = 3
adam = Adam(lr = 0.001 , beta_1=0.9)
sgd = SGD(lr=0.001,momentum=0.8)


weights = [  4.34558461   ,2.97682037   ,3.92124661   ,5.67350328 ,374.0300152 ]
print('='*60)
print(weights)
loss = weighted_categorical_crossentropy(weights)
if args.multitasking:
    weighted_cross_entropy = weighted_categorical_crossentropy(weights)
    cross_entropy = "categorical_crossentropy"
    tanimoto = Tanimoto_dual_loss()

if args.resunet_a == True:

    if args.multitasking:
        print('Multitasking enabled!')
        resuneta = Resunet_a2((rows, cols, channels), number_class, args)
        model = resuneta.model
        model.summary()
        # losses = {
        # 	"segmentation": weighted_cross_entropy,
        # 	"boundary": weighted_cross_entropy,
        #     "distance": weighted_cross_entropy,
        #     "color": cross_entropy,
        # }
        losses = {
        	"segmentation": tanimoto,
        	"boundary": tanimoto,
            "distance": tanimoto,
            "color": tanimoto,
        }
        lossWeights = {"segmentation": 1.0, "boundary": 1.0, "distance": 1.0,
        "color": 1.0}
        if args.gpu_parallel:
            with strategy.scope():
                model.compile(optimizer=adam, loss=losses, loss_weights=lossWeights, metrics=['accuracy'])
        else:
            model.compile(optimizer=adam, loss=losses, loss_weights=lossWeights, metrics=['accuracy'])
    else:
        resuneta = Resunet_a2((rows, cols, channels), number_class, args)
        model = resuneta.model
        model.summary()
        model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])

    print('ResUnet-a compiled!')
else:
    model = unet((rows, cols, channels), number_class)
    model.summary()

    model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
    #model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


filepath = './models/'
# define early stopping callback
# earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='min')
# checkpoint = ModelCheckpoint(filepath+'unet_exp_'+str(exp)+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [earlystop, checkpoint]

# train the model
if args.multitasking:
    start_training = time.time()
    # model_info = model.fit(x=patches_tr, y=y_fit, batch_size=batch_size, epochs=100, callbacks=callbacks_list, verbose=2, validation_data= (patches_val, val_fit) )
    model_info = model.fit(x=train_generator, epochs=100, callbacks=callbacks_list, verbose=2, validation_data=val_generator )
    end_training = time.time() - start_time
else:
    start_training = time.time()
    x_shape_batch = (batch_size, patch_size, patch_size, 3)
    y_shape_batch = (batch_size, patch_size, patch_size, 5)

    Train_model(model, patches_train, patches_tr_lb_h, patches_val, patches_val_lb_h, batch_size, epochs, patience=10, delta=0.001, x_shape_batch=x_shape_batch, y_shape_batch=y_shape_batch, seed=seed)
    # model_info = model.fit(x=train_generator, epochs=100, callbacks=callbacks_list, verbose=2, validation_data= val_generator)

    end_training = time.time() - start_time

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
