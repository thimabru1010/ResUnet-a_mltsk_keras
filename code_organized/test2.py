import time
from utils import np, plt, load_tiff_image, load_SAR_image, compute_metrics, data_augmentation, unet, normalization, \
RGB_image, extract_patches, patch_tiles, bal_aug_patches, extrac_patch2, test_FCN, pred_recostruction, \
weighted_categorical_crossentropy, mask_no_considered, tf, Adam, prediction, load_model, confusion_matrix, \
EarlyStopping, ModelCheckpoint, identity_block, ResNet50, color_map
import os
from utils2 import patch_tiles2, bal_aug_patches2, bal_aug_patches3, patch_tiles3, prediction2, output_prediction_FC, patch_tiles_prediction

import argparse

import gc
print('[DEBUG]')
gc.set_debug(gc.DEBUG_SAVEALL)
print(gc.get_count())

filepath = './models_new/'
exp=1
root_path = 'dataset'
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
#plt.imshow(final_mask)

print('[DEBUG]')
print(gc.get_count())
del img_mask_ref
gc.collect()
print(gc.get_count())

# Mask with tiles
# Divide tiles in 5 rows and 3 columns. Total = 15 tiles
tile_number = np.ones((1220,2200))
mask_c_1 = np.concatenate((tile_number, 2*tile_number, 3*tile_number), axis=1)
mask_c_2 = np.concatenate((4*tile_number, 5*tile_number, 6*tile_number), axis=1)
mask_c_3 = np.concatenate((7*tile_number, 8*tile_number, 9*tile_number), axis=1)
mask_c_4 = np.concatenate((10*tile_number, 11*tile_number, 12*tile_number), axis=1)
mask_c_5 = np.concatenate((13*tile_number, 14*tile_number, 15*tile_number), axis=1)
mask_tiles = np.concatenate((mask_c_1, mask_c_2, mask_c_3 , mask_c_4, mask_c_5), axis=0)

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
area = 11
# Prediction
patch_size = 64
ts_tiles = [ts1, ts2, ts3, ts4, ts5, ts6, ts7, ts8, ts9]
patches_test, patches_test_ref = patch_tiles_prediction(ts_tiles, mask_ts_, image_array, image_ref, None, patch_size, stride=patch_size)


result = model.predict(patches_test)
patches_pred = np.argmax(result, axis=-1)

true_labels = np.reshape(patches_test_ref, (patches_test_ref.shape[0]* patches_test_ref.shape[1]*patches_test_ref.shape[2]))
predicted_labels = np.reshape(patches_pred, (patches_pred.shape[0]* patches_pred.shape[1]*patches_pred.shape[2]))

# Metrics
metrics = compute_metrics(true_labels,predicted_labels)
cm = confusion_matrix(true_labels, predicted_labels)

print('Confusion  matrix \n', cm)
print()
print('Accuracy: ', metrics[0])
print('F1score: ', metrics[1])
print('Recall: ', metrics[2])
print('Precision: ', metrics[3])

# ref_final, pre_final, prob_recontructed, ref_reconstructed, mask_no_considered_, mask_ts, time_ts = prediction2(model, image_array, image_ref, final_mask, mask_ts_, patch_size, area)
# prob_recontructed, time_ts = output_prediction_FC(model, image_array, final_mask, patch_size)
# prob_recontructed, end_test = output_prediction_FC(model, image_array, final_mask, patch_size)

#%% Calculation of metrics

# ref1 = np.ones_like(image_ref).astype(np.float32)
# ref1 [image_ref == 2] = 0
# TileMask = mask_ts_ * ref1
# GTTruePositives = image_ref == 1
#
# Npoints = 100
# Pmax = np.max(mean_prob[GTTruePositives * TileMask == 1])
# ProbList = np.linspace(Pmax,0,Npoints)
# print(ProbList)
# #area = 620
# metrics_all = []
# #
# # for tm in range (0, 10):
# #     print(tm)
# #     prob_map = np.load(dirProbs+'/'+'prob_'+str(tm)+'.npy')
#
# metrics = matrics_AA_recall(ProbList, prob_recontructed, image_ref, mask_ts_, area)
# metrics_all.append(metrics)
#
# metrics_ = np.asarray(metrics_all)

# # Metrics
# cm = confusion_matrix(ref_final, pre_final)
#metrics = compute_metrics(ref_final, pre_final)
# print('Confusion  matrix \n', cm)
# print('Accuracy: ', metrics[0])
# print('F1score: ', metrics[1])
# print('Recall: ', metrics[2])
# print('Precision: ', metrics[3])
#
# # Alarm area
# total = (cm[1,1]+cm[0,1])/len(ref_final)*100
# print('Area to be analyzed',total)
#
# print('training time', end_training)
# print('test time', time_ts)
#
# #%% Show the results
# # prediction of the whole image
# fig1 = plt.figure('whole prediction')
# plt.imshow(prob_recontructed)
# # Show the test tiles
# fig2 = plt.figure('prediction of test set')
# plt.imshow(prob_recontructed*mask_ts)
