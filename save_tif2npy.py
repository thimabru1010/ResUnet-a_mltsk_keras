import numpy as np
from osgeo import ogr, gdal
import os
import cv2

# Functions
def load_tiff_image(patch):
  # Read tiff Image
   print (patch)
   gdal_header = gdal.Open(patch)
   img = gdal_header.ReadAsArray()
   return img

# Load images --- Mabel
# root_path = './'
# img_t1 = load_tiff_image(root_path+'images/18_08_2017_image'+'.tif').astype(np.float32)
# np.save('dataset_npy/18_08_2017_image_float32.npy', img_t1)
#
# img_t2 = load_tiff_image(root_path+'images/21_08_2018_image'+'.tif').astype(np.float32)
# np.save('dataset_npy/21_08_2018_image_float32.npy', img_t2)
#
# image_ref1 = load_tiff_image(root_path+'images/REFERENCE_2018_EPSG4674'+'.tif')
# np.save('dataset_npy/REFERENCE_2018_EPSG4674.npy', image_ref1)
#
# past_ref1 = load_tiff_image(root_path+'images/PAST_REFERENCE_FOR_2018_EPSG4674'+'.tif')
# np.save('dataset_npy/PAST_REFERENCE_FOR_2018_EPSG4674.npy', past_ref1)


# Dataset TCC
# root_path = 'dataset'
# img_t1_path = 'clipped_raster_004_66_2018.tif'
# img_t2_path = 'clipped_raster_004_66_2019.tif'
#
# # Load images
# img_t1 = load_tiff_image(os.path.join(root_path,img_t1_path)).astype(np.float32)
# np.save('dataset_npy/clipped_raster_004_66_2018.npy', img_t1)
#
# img_t2 = load_tiff_image(os.path.join(root_path,img_t2_path)).astype(np.float32)
# np.save('dataset_npy/clipped_raster_004_66_2019.npy', img_t2)
#
# img_mask_ref_path = 'mask_ref.tif'
# img_mask_ref = load_tiff_image(os.path.join(root_path, img_mask_ref_path))
# np.save('dataset_npy/mask_ref.npy', img_mask_ref)
#
# # Load deforastation reference
# image_ref = load_tiff_image(os.path.join(root_path,'labels/binary_clipped_2019.tif'))
# np.save(os.path.join('dataset_npy','labels/binary_clipped_2019.npy'), image_ref)
#
# # Load past deforastation reference
# past_ref1 = load_tiff_image(os.path.join(root_path,'labels/binary_clipped_2013_2018.tif'))
# np.save(os.path.join('dataset_npy','labels/binary_clipped_2013_2018.npy'), past_ref1)
#
# past_ref2 = load_tiff_image(os.path.join(root_path,'labels/binary_clipped_1988_2012.tif'))
# np.save(os.path.join('dataset_npy','labels/binary_clipped_1988_2012.tif'), past_ref2)
from skimage.transform import resize
#bottle_resized = resize(bottle, (140, 54))
# Homework 3
img_train = load_tiff_image('DATASETS/homework3/Image_Train.tif')
img_train = resize(img_train, (4500, 4000))
img_train = cv2.imencode(".jpeg", img_train)
cv2.imwrite('img_train.jpeg', img_train)
#np.save('homework3_npy/Image_Train.jpg', img_train)

del img_train

ref_train = load_tiff_image('DATASETS/homework3/Reference_Train.tif')
ref_train = resize(ref_train, (4500, 4000))
ref_train = cv2.imencode(".jpeg", ref_train)
cv2.imwrite('img_ref.jpeg', ref_train)
#np.save('homework3_npy/Reference_Train.jpg', ref_train)

# img_test = load_tiff_image('homework3/Image_Test.tif').astype(np.float32)
# np.save('homework3_npy/Image_Test.npy', img_test)
#
# ref_test = load_tiff_image('homework3/Reference_Test.tif')
# np.save('homework3_npy/Reference_Test.npy', ref_test)
