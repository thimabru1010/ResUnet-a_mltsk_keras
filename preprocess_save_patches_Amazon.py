from utils import np, load_npy_image, data_augmentation, mask_no_considered
import tensorflow as tf

from multitasking_utils import get_boundary_label, get_distance_label
import argparse
import os

from skimage.util.shape import view_as_windows

import gc
import psutil
import cv2
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def extract_patches(image, reference, patch_size, stride):
    window_shape = patch_size
    window_shape_array = (window_shape, window_shape, image.shape[2])
    window_shape_ref = (window_shape, window_shape)
    patches_array = np.array(view_as_windows(image,
                                             window_shape_array, step=stride))

    patches_ref = np.array(view_as_windows(reference,
                                           window_shape_ref, step=stride))

    print('Patches extraidos')
    print(patches_array.shape)
    num_row, num_col, p, row, col, depth = patches_array.shape

    print('fazendo reshape')
    patches_array = patches_array.reshape(num_row*num_col, row, col, depth)
    print(patches_array.shape)
    patches_ref = patches_ref.reshape(num_row*num_col, row, col)
    print(patches_ref.shape)

    return patches_array, patches_ref


def binarize_matrix(img_train_ref, label_dict):
    # Create binarized matrix
    w = img_train_ref.shape[0]
    h = img_train_ref.shape[1]
    # c = img_train_ref.shape[2]
    # binary_img_train_ref = np.zeros((1,w,h))
    binary_img_train_ref = np.full((w, h), -1, dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            r = img_train_ref[i][j][0]
            g = img_train_ref[i][j][1]
            b = img_train_ref[i][j][2]
            rgb = (r, g, b)
            rgb_key = str(rgb)
            binary_img_train_ref[i][j] = label_dict[rgb_key]

    return binary_img_train_ref


def normalize_rgb(img, norm_type=1):
    # OBS: Images need to be converted to before float32 to be normalized
    # TODO: Maybe should implement normalization with StandardScaler
    # Normalize image between [0, 1]
    if norm_type == 1:
        img /= 255.
    # Normalize image between [-1, 1]
    elif norm_type == 2:
        img /= 127.5 - 1.
    elif norm_type == 3:
        image_reshaped = img.reshape((img.shape[0]*img.shape[1]), img.shape[2])
        scaler = StandardScaler()
        scaler = scaler.fit(image_reshaped)
        image_normalized = scaler.fit_transform(image_reshaped)
        img = image_normalized.reshape(img.shape[0], img.shape[1], img.shape[2])

    return img


def normalize_hsv(img, norm_type=1):
    # OBS: Images need to be converted to before float32 to be normalized
    # TODO: Maybe should implement normalization with StandardScaler
    # Normalize image between [0, 1]
    if norm_type == 1:
        img[:, :, 0] /= 179.
        img[:, :, 1] /= 255.
        img[:, :, 2] /= 255.
    # Normalize image between [-1, 1]
    elif norm_type == 2:
        img[:, :, 0] /= 89.5 - 1.
        img[:, :, 1] /= 127.5 - 1.
        img[:, :, 2] /= 127.5 - 1.
    elif norm_type == 3:
        image_reshaped = img.reshape((img.shape[0]*img.shape[1]), img.shape[2])
        scaler = StandardScaler()
        scaler = scaler.fit(image_reshaped)
        image_normalized = scaler.fit_transform(image_reshaped)
        img = image_normalized.reshape(img.shape[0], img.shape[1], img.shape[2])

    return img

def count_deforastation(image_ref, image_mask_ref):
    total_no_def = 0
    total_def = 0

    # Make this to count the deforastation area
    image_ref[img_mask_ref == -99] = -1

    total_no_def += len(image_ref[image_ref == 0])
    total_def += len(image_ref[image_ref == 1])
    # Print number of samples of each class
    print('Total no-deforestaion class is {}'.format(len(image_ref[image_ref == 0])))
    print('Total deforestaion class is {}'.format(len(image_ref[image_ref == 1])))
    print('Percentage of deforestaion class is {:.2f}'.format((len(image_ref[image_ref == 1])*100)/len(image_ref[image_ref == 0])))

    image_ref[img_mask_ref == -99] = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm_type",
                        help="Choose type of normalization to be used",
                        type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--patch_size",
                        help="Choose size of patches", type=int, default=256)
    parser.add_argument("--stride",
                        help="Choose stride to be using on patches extraction",
                        type=int, default=32)
    parser.add_argument("--num_classes",
                        help="Choose number of classes to convert \
                        labels to one hot encoding", type=int, default=5)
    parser.add_argument("--data_aug",
                        help="Choose number of classes to convert \
                        labels to one hot encoding",
                        type=str2bool, default=True)
    parser.add_argument("--def_percent",
                        help="Choose minimum percentage of Deforastation",
                        type=int, default=5)
    args = parser.parse_args()

    print('='*50)
    print('Parameters')
    print(f'patch size={args.patch_size}')
    print(f'stride={args.stride}')
    print(f'Number of classes={args.num_classes} ')
    print(f'Norm type: {args.norm_type}')
    print(f'Using data augmentation? {args.data_aug}')
    print('='*50)

    root_path = './DATASETS/Amazon_npy'
    # Load images --------------------------------------------------------------
    img_t1_path = 'clipped_raster_004_66_2018.npy'
    img_t2_path = 'clipped_raster_004_66_2019.npy'
    img_t1 = load_npy_image(os.path.join(root_path, img_t1_path))
    img_t2 = load_npy_image(os.path.join(root_path, img_t2_path))

    # Convert shape from C x H x W --> H x W x C
    img_t1 = img_t1.transpose((1, 2, 0))
    img_t2 = img_t2.transpose((1, 2, 0))
    # img_train_normalized = normalization(img_train)
    print('Image 7 bands')
    print(img_t1.shape)
    print(img_t2.shape)

    # Concatenation of images
    image_array1 = np.concatenate((img_t1, img_t2), axis=-1)
    image_array1 = image_array1[:6100, :6600]
    h_, w_, channels = image_array1.shape
    print(f"Input image shape: {image_array1.shape}")

    # Load Mask area -----------------------------------------------------------
    # Mask constains exactly location of region since the satelite image
    # doesn't fill the entire resolution (Kinda rotated with 0 around)
    img_mask_ref_path = 'mask_ref.npy'
    img_mask_ref = load_npy_image(os.path.join(root_path, img_mask_ref_path))
    img_mask_ref = img_mask_ref[:6100, :6600]
    print(f"Mask area reference shape: {img_mask_ref.shape}")

    # Load deforastation reference ---------------------------------------------
    '''
        0 --> No deforastation
        1 --> Deforastation
    '''
    image_ref = load_npy_image(os.path.join(root_path,
                                            'labels/binary_clipped_2019.npy'))
    # Clip to fit tiles of your specific image
    image_ref = image_ref[:6100, :6600]
    # image_ref[img_mask_ref == -99] = -1
    print(f"Image reference shape: {image_ref.shape}")

    # Load past deforastation reference ----------------------------------------
    past_ref1 = load_npy_image(os.path.join(root_path,
                                            'labels/binary_clipped_2013_2018.npy'))
    past_ref2 = load_npy_image(os.path.join(root_path,
                                            'labels/binary_clipped_1988_2012.npy'))
    past_ref_sum = past_ref1 + past_ref2
    # Clip to fit tiles of your specific image
    past_ref_sum = past_ref_sum[:6100, :6600]
    # past_ref_sum[img_mask_ref==-99] = -1
    # Doing the sum, there are some pixels with value 2 (Case when both were deforastation).
    # past_ref_sum[past_ref_sum == 2] = 1
    # Same thing for background area (different from no deforastation)
    # past_ref_sum[past_ref_sum==-2] = -1
    print(f"Past reference shape: {past_ref_sum.shape}")

    #  Creation of buffer
    buffer = 2
    # Gather past deforestation with actual deforastation
    '''
        0 --> No deforastation
        1 --> Deforastation
        2 --> Past deforastation (No considered)
    '''
    final_mask = mask_no_considered(image_ref, buffer, past_ref_sum)
    # final_mask[img_mask_ref == -99] = -1

    unique, counts = np.unique(final_mask, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    print(f'Pixels of final mask: {counts_dict}')
    # Calculates weights for weighted cross entropy
    total_pixels = counts_dict[0] + counts_dict[1] + counts_dict[2]
    weight0 = total_pixels / counts_dict[0]
    weight1 = total_pixels / counts_dict[1]

    del img_t1, img_t2, image_ref, past_ref1, past_ref2

    count_deforastation(image_ref, img_mask_ref)

    # Mask with tiles
    # Divide tiles in 5 rows and 3 columns. Total = 15 tiles
    # tile.shape --> (6100/5, 6600/3) = (1220, 2200)
    tile_number = np.ones((1220, 2200))
    mask_c_1 = np.concatenate((tile_number, 2*tile_number, 3*tile_number), axis=1)
    mask_c_2 = np.concatenate((4*tile_number, 5*tile_number, 6*tile_number), axis=1)
    mask_c_3 = np.concatenate((7*tile_number, 8*tile_number, 9*tile_number), axis=1)
    mask_c_4 = np.concatenate((10*tile_number, 11*tile_number, 12*tile_number), axis=1)
    mask_c_5 = np.concatenate((13*tile_number, 14*tile_number, 15*tile_number), axis=1)
    mask_tiles = np.concatenate((mask_c_1, mask_c_2, mask_c_3, mask_c_4, mask_c_5), axis=0)

    mask_tr_val = np.zeros((mask_tiles.shape))
    tr1 = 5
    tr2 = 8
    tr3 = 10
    tr4 = 13
    val1 = 7
    val2 = 10
    # tr5 = 7
    # tr6 = 10

    mask_tr_val[mask_tiles == tr1] = 1
    mask_tr_val[mask_tiles == tr2] = 1
    mask_tr_val[mask_tiles == tr3] = 1
    mask_tr_val[mask_tiles == tr4] = 1
    mask_tr_val[mask_tiles == val1] = 2
    mask_tr_val[mask_tiles == val2] = 2
    # mask_tr_val[mask_tiles == tr5] = 1
    # mask_tr_val[mask_tiles == tr6] = 1


    # stride = patch_size
    patches_tr, patches_tr_ref = extract_patches(img_train,
                                                 binary_img_train_ref,
                                                 args.patch_size, args.stride)
    print('patches extracted!')
    process = psutil.Process(os.getpid())
    print('[CHECKING MEMORY]')
    # print(process.memory_info().rss)
    print(process.memory_percent())
    del binary_img_train_ref, img_t1, img_t2
    # print(process.memory_info().rss)
    print(process.memory_percent())
    gc.collect()
    print('[GC COLLECT]')
    print(process.memory_percent())

    print('saving images...')
    folder_path = f'./DATASETS/patch_size={args.patch_size}_' + \
                f'stride={args.stride}_norm_type={args.norm_type}_data_aug={args.data_aug}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        os.makedirs(os.path.join(folder_path, 'train'))
        os.makedirs(os.path.join(folder_path, 'labels'))
        os.makedirs(os.path.join(folder_path, 'labels/seg'))
        os.makedirs(os.path.join(folder_path, 'labels/bound'))
        os.makedirs(os.path.join(folder_path, 'labels/dist'))
        os.makedirs(os.path.join(folder_path, 'labels/color'))


    def filename(i):
        return f'patch_{i}.npy'


    print(f'Number of patches: {len(patches_tr)}')
    if args.data_aug:
        print(f'Number of patches expected: {len(patches_tr)*5}')
    for i in tqdm(range(len(patches_tr))):
        if args.data_aug:
            img_aug, label_aug = data_augmentation(patches_tr[i], patches_tr_ref[i])
        else:
            img_aug, label_aug = np.expand_dims(patches_tr[i], axis=0), np.expand_dims(patches_tr_ref[i], axis=0)
        label_aug_h = tf.keras.utils.to_categorical(label_aug, args.num_classes)
        for j in range(len(img_aug)):
            # Input image RGB
            # Float32 its need to train the model
            img_float = img_aug[j].astype(np.float32)
            img_normalized = normalize_rgb(img_float, norm_type=args.norm_type)
            np.save(os.path.join(folder_path, 'train', filename(i*5 + j)),
                    img_normalized)
            # All multitasking labels are saved in one-hot
            # Segmentation
            np.save(os.path.join(folder_path, 'labels/seg', filename(i*5 + j)),
                    label_aug_h[j])
            # Boundary
            bound_label_h = get_boundary_label(label_aug_h[j])
            np.save(os.path.join(folder_path, 'labels/bound', filename(i*5 + j)),
                    bound_label_h)
            # Distance
            dist_label_h = get_distance_label(label_aug_h[j])
            np.save(os.path.join(folder_path, 'labels/dist', filename(i*5 + j)),
                    dist_label_h)
            # Color
            hsv_patch = cv2.cvtColor(img_aug[j],
                                     cv2.COLOR_RGB2HSV).astype(np.float32)
            # Float32 its need to train the model
            hsv_patch = normalize_hsv(hsv_patch, norm_type=args.norm_type)
            np.save(os.path.join(folder_path, 'labels/color', filename(i*5 + j)),
                    hsv_patch)
