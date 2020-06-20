import numpy as np
from osgeo import ogr, gdal

# Functions
def load_tiff_image(patch):
  # Read tiff Image
   print (patch)
   gdal_header = gdal.Open(patch)
   img = gdal_header.ReadAsArray()
   return img

# Load images
root_path = './'
img_t1 = load_tiff_image(root_path+'images/18_08_2017_image'+'.tif').astype(np.float32)
np.save('dataset_npy/18_08_2017_image_float32.npy', img_t1)

img_t2 = load_tiff_image(root_path+'images/21_08_2018_image'+'.tif').astype(np.float32)
np.save('dataset_npy/21_08_2018_image_float32.npy', img_t2)

image_ref1 = load_tiff_image(root_path+'images/REFERENCE_2018_EPSG4674'+'.tif')
np.save('dataset_npy/REFERENCE_2018_EPSG4674.npy', image_ref1)

past_ref1 = load_tiff_image(root_path+'images/PAST_REFERENCE_FOR_2018_EPSG4674'+'.tif')
np.save('dataset_npy/PAST_REFERENCE_FOR_2018_EPSG4674.npy', past_ref1)
