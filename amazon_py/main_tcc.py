import time
from utils import np, plt, load_tiff_image, load_SAR_image, compute_metrics, data_augmentation, unet, normalization, \
RGB_image, extract_patches, patch_tiles, bal_aug_patches, extrac_patch2, test_FCN, pred_recostruction, \
weighted_categorical_crossentropy, mask_no_considered, tf, Adam, prediction, load_model, confusion_matrix, \
EarlyStopping, ModelCheckpoint, identity_block, ResNet50, color_map, load_npy_image
import os
from utils2 import patch_tiles2, bal_aug_patches2, bal_aug_patches3, patch_tiles3

from ResUnet_a.model import Resunet_a
from ResUnet_a.model2 import Resunet_a2
from sklearn.model_selection import train_test_split

import argparse

import gc
# gc.set_debug(gc.DEBUG_SAVEALL)
# print(gc.get_count())

parser = argparse.ArgumentParser()
parser.add_argument("--resunet_a",
    help="choose resunet-a model or not", type=int, default=0)
parser.add_argument("--multitasking",
    help="choose resunet-a model or not", type=int, default=0)
args = parser.parse_args()

root_path = './DATASETS/dataset_npy'
img_t1_path = 'clipped_raster_004_66_2018.npy'
img_t2_path = 'clipped_raster_004_66_2019.npy'

# Load images
img_t1 = load_npy_image(os.path.join(root_path,img_t1_path)).astype(np.float32)
img_t1 = img_t1.transpose((1,2,0))
img_t2 = load_npy_image(os.path.join(root_path,img_t2_path)).astype(np.float32)
img_t2 = img_t2.transpose((1,2,0))

# Concatenation of images
image_array1 = np.concatenate((img_t1, img_t2), axis = -1).astype(np.float32)
image_array1 = image_array1[:6100,:6600]
h_, w_, channels = image_array1.shape
print(f"Input image shape: {image_array1.shape}")

# Normalization
type_norm = 1
image_array = normalization(image_array1, type_norm)
#print(np.min(image_array), np.max(image_array))

# Load Mask area
img_mask_ref_path = 'mask_ref.npy'
img_mask_ref = load_npy_image(os.path.join(root_path, img_mask_ref_path))
img_mask_ref = img_mask_ref[:6100, :6600]
print(f"Mask area reference shape: {img_mask_ref.shape}")

# Load deforastation reference
image_ref = load_npy_image(os.path.join(root_path,
                                        'labels/binary_clipped_2019.npy'))
# Clip to fit tiles of your specific image
image_ref = image_ref[:6100, :6600]
image_ref[img_mask_ref == -99] = -1
print(f"Image reference shape: {image_ref.shape}")

# Load past deforastation reference
past_ref1 = load_npy_image(os.path.join(root_path, 'labels/binary_clipped_2013_2018.npy'))
past_ref2 = load_npy_image(os.path.join(root_path, 'labels/binary_clipped_1988_2012.npy'))
past_ref_sum = past_ref1 + past_ref2
# Clip to fit tiles of your specific image
past_ref_sum = past_ref_sum[:6100, :6600]
#past_ref_sum[img_mask_ref==-99] = -1
# Doing the sum, there are some pixels with value 2 (Case when both were deforastation).
past_ref_sum[past_ref_sum == 2] = 1
# Same thing for background area (different from no deforastation)
#past_ref_sum[past_ref_sum==-2] = -1
print(f"Past reference shape: {past_ref_sum.shape}")

#  Creation of buffer
buffer = 2
final_mask = mask_no_considered(image_ref, buffer, past_ref_sum)
final_mask[img_mask_ref==-99] = -1

unique, counts = np.unique(final_mask, return_counts=True)
counts_dict = dict(zip(unique, counts))
print(f'Pixels of final mask: {counts_dict}')
total_pixels = counts_dict[0] + counts_dict[1] + counts_dict[2]
weight0 = total_pixels / counts_dict[0]
weight1 = total_pixels / counts_dict[1]
final_mask[img_mask_ref==-99] = 0

# Mask with tiles
# Divide tiles in 5 rows and 3 columns. Total = 15 tiles
tile_number = np.ones((1220, 2200))
mask_c_1 = np.concatenate((tile_number, 2*tile_number, 3*tile_number), axis=1)
mask_c_2 = np.concatenate((4*tile_number, 5*tile_number, 6*tile_number), axis=1)
mask_c_3 = np.concatenate((7*tile_number, 8*tile_number, 9*tile_number), axis=1)
mask_c_4 = np.concatenate((10*tile_number, 11*tile_number, 12*tile_number), axis=1)
mask_c_5 = np.concatenate((13*tile_number, 14*tile_number, 15*tile_number), axis=1)
mask_tiles = np.concatenate((mask_c_1, mask_c_2, mask_c_3 , mask_c_4, mask_c_5), axis=0)

mask_tr_val = np.zeros((mask_tiles.shape))
tr1 = 5
tr2 = 8
tr3 = 10
tr4 = 13
# val1 = 7
# val2 = 10
tr5 = 7
tr6 = 10

mask_tr_val[mask_tiles == tr1] = 1
mask_tr_val[mask_tiles == tr2] = 1
mask_tr_val[mask_tiles == tr3] = 1
mask_tr_val[mask_tiles == tr4] = 1
# mask_tr_val[mask_tiles == val1] = 2
# mask_tr_val[mask_tiles == val2] = 2
mask_tr_val[mask_tiles == tr5] = 1
mask_tr_val[mask_tiles == tr6] = 1

total_no_def = 0
total_def = 0

# Make this to count the deforastation area
image_ref[img_mask_ref==-99] = -1

total_no_def += len(image_ref[image_ref==0])
total_def += len(image_ref[image_ref==1])
# Print number of samples of each class
print('Total no-deforestaion class is {}'.format(len(image_ref[image_ref==0])))
print('Total deforestaion class is {}'.format(len(image_ref[image_ref==1])))
print('Percentage of deforestaion class is {:.2f}'.format((len(image_ref[image_ref==1])*100)/len(image_ref[image_ref==0])))

#image_ref[img_mask_ref==-99] = 0
#%% Patches extraction
patch_size = 128
#stride = patch_size
stride = patch_size//8

print("="*40)
print(f'Patche size: {patch_size}')
print(f'Stride: {stride}')
print("="*40)

# Percent of class deforestation
percent = 5
# 0 -> No-DEf, 1-> Def, 2 -> No considered
number_class = 3

# Trainig tiles
print('extracting training patches....')
tr_tiles = [tr1, tr2, tr3, tr4, tr5, tr6]
final_mask[img_mask_ref==-99] = -1
#test = list(range(1,16))
# patches_tr, patches_tr_ref = patch_tiles3(test, mask_tiles, image_array, final_mask, patch_size, stride)
# print(patches_tr.shape)
# print(patches_tr_ref.shape)
patches_tr, patches_tr_ref = patch_tiles2(tr_tiles, mask_tiles, image_array, final_mask, img_mask_ref, patch_size, stride, percent)

print(f"Trainig patches size: {patches_tr.shape}")
print(f"Trainig ref patches size: {patches_tr_ref.shape}")

patches_tr_aug, patches_tr_ref_aug = bal_aug_patches2(percent, patch_size, patches_tr, patches_tr_ref)

print(f"Trainig patches size with data aug: {patches_tr_aug.shape}")
print(f"Trainig ref patches sizewith data aug: {patches_tr_ref_aug.shape}")

patches_tr_ref_aug_h = tf.keras.utils.to_categorical(patches_tr_ref_aug, number_class)

# Validation tiles
print('extracting validation patches....')
#Validation train_test_split
patches_tr_aug, patches_val_aug, patches_tr_ref_aug_h, patches_val_ref_aug_h   = train_test_split(patches_tr_aug, patches_tr_ref_aug_h, test_size=0.2, random_state=42)
# val_tiles = [val1, val2]
# # patches_val, patches_val_ref = patch_tiles(val_tiles, mask_tiles, image_array, final_mask, patch_size, stride)
# patches_val, patches_val_ref = patch_tiles2(val_tiles, mask_tiles, image_array, final_mask, img_mask_ref, patch_size, stride, percent)
#
# print(f"Validation patches size: {patches_val.shape}")
# print(f"Validation ref patches size: {patches_val_ref.shape}")
#
# patches_val_aug, patches_val_ref_aug = bal_aug_patches2(percent, patch_size, patches_val, patches_val_ref)
# patches_val_ref_aug_h = tf.keras.utils.to_categorical(patches_val_ref_aug, number_class)

print(f"Validation patches size with data aug: {patches_val_aug.shape}")
print(f"Validation ref patches sizewith data aug: {patches_val_ref_aug_h.shape}")

#%%
start_time = time.time()
exp = 1
rows = patch_size
cols = patch_size
adam = Adam(lr = 0.0001 , beta_1=0.9)
batch_size = 8

weights = [weight0, weight1, 0]
print(f"Weights: {weights}")
#print('='*80)
#print(gc.get_count())
loss = weighted_categorical_crossentropy(weights)
if args.resunet_a == True:
    '''
        model already compiled
    '''
    #model = Resunet_a((channels, cols, rows))
    resuneta = Resunet_a2((rows, cols, channels), number_class, args)
    model = resuneta.model
    print('ResUnet-a compiled!')
else:
    model = unet((rows, cols, channels))
    #model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # print model information
    model.summary()

filepath = './models/'
# define early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath+'unet_exp_'+str(exp)+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [earlystop, checkpoint]

# train the model
start_training = time.time()
model_info = model.fit(patches_tr_aug, patches_tr_ref_aug_h, batch_size=batch_size, epochs=10, callbacks=callbacks_list, verbose=2, validation_data= (patches_val_aug, patches_val_ref_aug_h) )
end_training = time.time() - start_time
#%% Test model
# Creation of mask with test tiles
mask_ts_ = np.zeros((mask_tiles.shape))
ts1 = 1
ts2 = 2
ts3 = 3
ts4 = 4
ts5 = 6
ts6 = 9
ts7 = 12
ts8 = 14
ts9 = 15
mask_ts_[mask_tiles == ts1] = 1
mask_ts_[mask_tiles == ts2] = 1
mask_ts_[mask_tiles == ts3] = 1
mask_ts_[mask_tiles == ts4] = 1
mask_ts_[mask_tiles == ts5] = 1
mask_ts_[mask_tiles == ts6] = 1
mask_ts_[mask_tiles == ts7] = 1
mask_ts_[mask_tiles == ts8] = 1
mask_ts_[mask_tiles == ts9] = 1

#% Load model
model = load_model(filepath+'unet_exp_'+str(exp)+'.h5', compile=False)
model.summary()
area = 11
# Prediction
ref_final, pre_final, prob_recontructed, ref_reconstructed, mask_no_considered_, mask_ts, time_ts = prediction(model, image_array, image_ref, final_mask, mask_ts_, patch_size, area)

# Metrics
cm = confusion_matrix(ref_final, pre_final)
metrics = compute_metrics(ref_final, pre_final)
print('Confusion  matrix \n', cm)
print('Accuracy: ', metrics[0])
print('F1score: ', metrics[1])
print('Recall: ', metrics[2])
print('Precision: ', metrics[3])

# Alarm area
total = (cm[1,1]+cm[0,1])/len(ref_final)*100
print('Area to be analyzed',total)

print('training time', end_training)
print('test time', time_ts)

#%% Show the results
# prediction of the whole image
fig1 = plt.figure('whole prediction')
plt.imshow(prob_recontructed)
plt.imsave('whole_pred.jpg', prob_recontructed)
# Show the test tiles
fig2 = plt.figure('prediction of test set')
plt.imshow(prob_recontructed*mask_ts)
plt.imsave('pred_test_set.jpg', prob_recontructed*mask_ts)
plt.show()
