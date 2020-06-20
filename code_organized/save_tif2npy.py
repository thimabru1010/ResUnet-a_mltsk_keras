import numpy as np
from osgeo import ogr, gdal
import os

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
root_path = 'dataset'
img_t1_path = 'clipped_raster_004_66_2018.tif'
img_t2_path = 'clipped_raster_004_66_2019.tif'

# Load images
img_t1 = load_tiff_image(os.path.join(root_path,img_t1_path)).astype(np.float32)
np.save('dataset_npy/clipped_raster_004_66_2018.npy', img_t1)

img_t2 = load_tiff_image(os.path.join(root_path,img_t2_path)).astype(np.float32)
np.save('dataset_npy/clipped_raster_004_66_2019.npy', img_t2)

img_mask_ref_path = 'mask_ref.tif'
img_mask_ref = load_tiff_image(os.path.join(root_path, img_mask_ref_path))
np.save('dataset_npy/mask_ref.npy', img_mask_ref)

# Load deforastation reference
image_ref = load_tiff_image(os.path.join(root_path,'labels/binary_clipped_2019.tif'))
np.save(os.path.join('dataset_npy','labels/binary_clipped_2019.npy'), image_ref)

# Load past deforastation reference
past_ref1 = load_tiff_image(os.path.join(root_path,'labels/binary_clipped_2013_2018.tif'))
np.save(os.path.join('dataset_npy','labels/binary_clipped_2013_2018.npy'), past_ref1)

past_ref2 = load_tiff_image(os.path.join(root_path,'labels/binary_clipped_1988_2012.tif'))
np.save(os.path.join('dataset_npy','labels/binary_clipped_1988_2012.tif'), past_ref2)
