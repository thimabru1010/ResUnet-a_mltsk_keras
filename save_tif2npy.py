import numpy as np
from osgeo import gdal
import os
import cv2
from skimage.transform import resize


def load_tiff_image(patch):
    # Read tiff Image
    print(patch)
    gdal_header = gdal.Open(patch)
    img = gdal_header.ReadAsArray()
    return img

# ISPRS
img_train = load_tiff_image('DATASETS/homework3/Image_Train.tif')
print(img_train.shape)
np.save('DATASETS/ISPRS_npy/Image_Train.npy', img_train)
print('img train saved')
del img_train

ref_train = load_tiff_image('DATASETS/homework3/Reference_Train.tif')
print(ref_train.shape)
np.save('DATASETS/ISPRS_npy/Reference_Train.npy', ref_train)
print('ref train saved')
del ref_train

img_test = load_tiff_image('DATASETS/homework3/Image_Test.tif')
np.save('DATASETS/ISPRS_npy/Image_Test.npy', img_test)
print('img test saved')
del img_test

ref_test = load_tiff_image('DATASETS/homework3/Reference_Test.tif')
np.save('DATASETS/ISPRS_npy/Reference_Test.npy', ref_test)
print('ref test saved')
