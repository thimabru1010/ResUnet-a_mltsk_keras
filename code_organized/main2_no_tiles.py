#import utils
import time
from utils import np, plt, load_tiff_image, load_SAR_image, compute_metrics, data_augmentation, unet, normalization, \
RGB_image, extract_patches, patch_tiles, bal_aug_patches, extrac_patch2, test_FCN, pred_recostruction, \
weighted_categorical_crossentropy, mask_no_considered, tf, Adam, prediction, load_model, confusion_matrix, \
EarlyStopping, ModelCheckpoint, identity_block, ResNet50, color_map
import os
from utils2 import patch_tiles2, bal_aug_patches2, bal_aug_patches3, extract_patches_right_region

import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
    help="dataset path", type=str, default='dataset')
args = parser.parse_args()

root_path = args.dataset
img_t1_path = 'clipped_raster_004_66_2018.tif'
img_t2_path = 'clipped_raster_004_66_2019.tif'

# Load images
img_t1 = load_tiff_image(os.path.join(root_path,img_t1_path)).astype(np.float32)
img_t1 = img_t1.transpose((1,2,0))
img_t2 = load_tiff_image(os.path.join(root_path,img_t2_path)).astype(np.float32)
img_t2 = img_t2.transpose((1,2,0))

# Concatenation of images
image_array1 = np.concatenate((img_t1, img_t2), axis = -1).astype(np.float32)
image_array1 = image_array1[:6100,:6600]
h_, w_, channels = image_array1.shape
print(f"Input image shape: {image_array1.shape}")

# Normalization
type_norm = 1
image_array = normalization(image_array1, type_norm)
print(np.min(image_array), np.max(image_array))

# Load Mask area
img_mask_ref_path = 'mask_ref.tif'
img_mask_ref = load_tiff_image(os.path.join(root_path, img_mask_ref_path))
img_mask_ref = img_mask_ref[:6100,:6600]
print(f"Mask area reference shape: {img_mask_ref.shape}")

# Load deforastation reference
image_ref = load_tiff_image(os.path.join(root_path,'labels/binary_clipped_2019.tif'))
# Clip to fit tiles of your specific image
image_ref = image_ref[:6100,:6600]
image_ref[img_mask_ref==-99] = -1
print(f"Image reference shape: {image_ref.shape}")

# Load past deforastation reference
past_ref1 = load_tiff_image(os.path.join(root_path,'labels/binary_clipped_2013_2018.tif'))
past_ref2 = load_tiff_image(os.path.join(root_path,'labels/binary_clipped_1988_2012.tif'))
past_ref_sum = past_ref1 + past_ref2
# Clip to fit tiles of your specific image
past_ref_sum = past_ref_sum[:6100,:6600]
#past_ref_sum[img_mask_ref==-99] = -1
# Doing the sum, there are some pixels with value 2 (Case when both were deforastation).
past_ref_sum[past_ref_sum==2] = 1
# Same thing for background area (different from no deforastation)
#past_ref_sum[past_ref_sum==-2] = -1
print(f"Past reference shape: {past_ref_sum.shape}")

#  Creation of buffer
buffer = 2
final_mask = mask_no_considered(image_ref, buffer, past_ref_sum)
final_mask[img_mask_ref==-99] = -1
unique, counts = np.unique(final_mask, return_counts=True)
counts_dict = dict(zip(unique, counts))
print(counts_dict)
total_pixels = counts_dict[0] + counts_dict[1] + counts_dict[2]
weight0 = total_pixels / counts_dict[0]
weight1 = total_pixels / counts_dict[1]
final_mask[img_mask_ref==-99] = 0

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

image_ref[img_mask_ref==-99] = 0
#%% Patches extraction
patch_size = 128
stride = patch_size//8
#stride = patch_size//1

print("="*40)
print(f'Patche size: {patch_size}')
print(f'Stride: {stride}')
print("="*40)

# Percent of class deforestation
percent = 5
# 0 -> No-DEf, 1-> Def, 2 -> No considered
number_class = 3

# Trainig
print('extracting training patches....')

patches_train, patches_train_ref =extract_patches_right_region(image_array, final_mask, img_mask_ref, patch_size, stride)
print(f'Number of patches: {len(patches_train)}, {len(patches_train_ref)}')

patches_tr, patches_val, patches_tr_ref, patches_val_ref = train_test_split(patches_train, patches_train_ref, test_size=0.2, random_state=42)

print(len(patches_tr))
print(np.asarray(patches_tr).shape)

patches_tr_aug, patches_tr_ref_aug = bal_aug_patches2(percent, patch_size, np.asarray(patches_tr), np.asarray(patches_tr_ref))
patches_tr_ref_aug_h = tf.keras.utils.to_categorical(patches_tr_ref_aug, number_class)

# Validation
print('extracting validation patches....')

patches_val_aug, patches_val_ref_aug = bal_aug_patches2(percent, patch_size, np.asarray(patches_val), np.asarray(patches_val_ref))
patches_val_ref_aug_h = tf.keras.utils.to_categorical(patches_val_ref_aug, number_class)

#%%
start_time = time.time()
exp = 1
rows = patch_size
cols = patch_size
adam = Adam(lr = 0.0001 , beta_1=0.9)
batch_size = 8

#weights = [0.5, 0.5, 0]
weights = [weight0, weight1, 0]
print(f"Weights: {weights}")
print('='*80)
loss = weighted_categorical_crossentropy(weights)
model = unet((rows, cols, channels))
#model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
# print model information
model.summary()
filepath = './models_new/'
# define early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath+'unet_exp_'+str(exp)+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [earlystop, checkpoint]
# train the model
start_training = time.time()
model_info = model.fit(patches_tr_aug, patches_tr_ref_aug_h, batch_size=batch_size, epochs=100, callbacks=callbacks_list, verbose=2, validation_data= (patches_val_aug, patches_val_ref_aug_h) )
end_training = time.time() - start_time
#%% Test model
# Creation of mask with test tiles
mask_ts_ = np.zeros((mask_tiles.shape))
ts1 = 2
ts2 = 3
ts3 = 4
ts4 = 8
ts5 = 9
ts6 = 10
ts7 = 11
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
# Show the test tiles
fig2 = plt.figure('prediction of test set')
plt.imshow(prob_recontructed*mask_ts)
