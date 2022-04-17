# ResUnet-a multitasking

This is a reimplementation of [ResUnet-a multitasking](https://github.com/feevos/resuneta) in keras. This repo was mainly build for Amazon deforestation, but can also be used in ISPRS dataset.

## Preprocess

Firstly, we need convert our image from tif to a numpy file to facilitate our work, specially if you are going to train in other machine, can be very difficult and boring to install GDAL and OSGEO. For this, you can use the script `save_tif2npy.py` to convert a tif to npy file. 

Then, we need to chop the image in patches to feed the neural network, because a remote sensing image has a big resolution. In our preprocessing step, we're going to convert to the specific channel format keras support, categorize pixels (e.g. convert to an integer number pixels in RGB. The new image will have a single channel), chop in pacthes and normalize each of the patches. In addition, the multitasking labels are also generated (boundary detection, distance transform and color transforation). Since I had very few images to train ResUnet-a, I applied augmentations (rot90, rot180, horizontal flip and vertical flip) to the patches and added them in the training set, multplying by 5 the size of dataset. To do this preprocessing step, use `preprocess_save_patches_ISPRS.py`

## Training

In training, remember to pass as arguments in the script the number of the patch size (`-ps, --patch_size`) in preprocessing step and number of classes of the dataset (`--num_classes`). For the multitasking case, I implemented tanimoto dual loss, used in the original paper, for tensorflow 2.x based on the [original mxnet implementation](https://github.com/feevos/resuneta/blob/master/nn/loss/loss.py). If you don't want to use tanimoto, You can use **Categorical Cross Entropy** for the *segmentation task*, **Binary Cross Entropy** for *boundary detection*, since it's a multi-label classification, and regression losses for color and *distance transform* like **MSE**. Also, You can choose in the arguments to use between weighted and standard cross entropy for segmentation task. The weights for each loss can be chosen in the arguments (`--bound_weight`, `--dist_weight`, `--color_weight`). The training is simplmented with **early stopping**. 

The dataset path can be chosen with the args `-dp` or `--dataset_path`. The logs and checkpoints will be saved in the `--results_path` folder. To see logs, you can run tensorboard into this log folder specified ``tensorboard --logdir log_folder_path``.

GPU parallel is supported and ca be eabled with `--gpu_parallel`.

## Testig

In Testig step, we have to chop the input image in patches to do the inference and then reconstruct the predicted images to compare with the original. Also, we can visualize each patch per class. For the color transformation in multitasking the prediction seen will be the RGB image converted from HSV and the difference normalized between the original and prediciton image in RGB. Besides the visualization, a confusion matrix, accuracy, F1-score, Recall and Precision will be calculated.
