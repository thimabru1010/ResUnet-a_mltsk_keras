# Import libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
#from osgeo import ogr, gdal
import os
from skimage.util.shape import view_as_windows
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Conv2D, MaxPool2D, Conv2DTranspose, Dropout, concatenate, \
Input, UpSampling2D, Flatten, GlobalAveragePooling2D, BatchNormalization, Add, ZeroPadding2D
from tensorflow.keras.models import Model, load_model, Sequential
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#from tensorflow.keras.applications.vgg16 import preprocess_input
from skimage.morphology import disk
from skimage.filters import rank
import skimage.morphology
from contextlib import redirect_stdout
import time
import tensorflow.keras.backend as K

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Functions
def load_tiff_image(patch):
  # Read tiff Image
   print (patch)
   gdal_header = gdal.Open(patch)
   img = gdal_header.ReadAsArray()
   return img

def load_npy_image(patch):
  # Read npy Image converted from tiff
   print (patch)
   img = np.load(patch)
   return img

def load_SAR_image(patch):
    # Read SAR Image
    print (patch)
    gdal_header = gdal.Open(patch)
    db_img = gdal_header.ReadAsArray()
    img = 10**(db_img/10)
    return img

def compute_metrics(true_labels, predicted_labels):
    accuracy = 100*accuracy_score(true_labels, predicted_labels)
    f1score = 100*f1_score(true_labels, predicted_labels, average=None)
    recall = 100*recall_score(true_labels, predicted_labels, average=None)
    precision = 100*precision_score(true_labels, predicted_labels, average=None)
    return accuracy, f1score, recall, precision

def extract_patches_mask_indices(input_image, patch_size, stride):
    h, w = input_image.shape
    image_indices = np.arange(h*w).reshape(h,w)
    window_shape = patch_size
    window_shape_array = (window_shape, window_shape)
    patches_array = np.array(view_as_windows(image_indices, window_shape_array, step = stride))
    num_row,num_col,row,col = patches_array.shape
    patches_array = patches_array.reshape(num_row*num_col,row,col)
    return patches_array

def data_augmentation(image, labels):
    aug_imgs = np.zeros((5, image.shape[0], image.shape[1], image.shape[2]))
    aug_lbs = np.zeros((5, image.shape[0], image.shape[1]))

    for i in range(0, len(aug_imgs)):
        aug_imgs[0, :, :, :] = image
        aug_imgs[1, :, :, :] = np.rot90(image, 1)
        aug_imgs[2, :, :, :] = np.rot90(image, 2)
        #aug_imgs[3, :, :, :] = np.rot90(image, 3)
        #horizontal_flip = np.flip(image,0)
        aug_imgs[3, :, :, :] = np.flip(image,0)
        aug_imgs[4, :, :, :] = np.flip(image, 1)
        #aug_imgs[6, :, :] = np.rot90(horizontal_flip, 2)
        #aug_imgs[7, :, :] =np.rot90(horizontal_flip, 3)

    for i in range(0, len(aug_lbs)):
        aug_lbs[0, :, :] = labels
        aug_lbs[1, :, :] = np.rot90(labels, 1)
        aug_lbs[2, :, :] = np.rot90(labels, 2)
        #aug_lbs[3, :, :] = np.rot90(labels, 3)
        #horizontal_flip_lb = np.flip(labels,0)
        aug_lbs[3, :, :] = np.flip(labels,0)
        aug_lbs[4, :, :] = np.flip(labels, 1)
        #aug_lbs[6, :, :] = np.rot90(horizontal_flip_lb, 2)
        #aug_lbs[7, :, :] =np.rot90(horizontal_flip_lb, 3)

    return aug_imgs, aug_lbs

# Original model
def unet(input_shape):
    input_img = Input(input_shape)

    f1 = 32
    conv1 = Conv2D(f1 , (3 , 3) , activation='relu' , padding='same', name = 'conv1')(input_img)
    pool1 = MaxPool2D((2 , 2))(conv1)

    conv2 = Conv2D(f1*2 , (3 , 3) , activation='relu' , padding='same', name = 'conv2')(pool1)
    pool2 = MaxPool2D((2 , 2))(conv2)

    conv3 = Conv2D(f1*4 , (3 , 3) , activation='relu' , padding='same', name = 'conv3')(pool2)
    pool3 = MaxPool2D((2 , 2))(conv3)

    conv4 = Conv2D(f1*8 , (3 , 3) , activation='relu' , padding='same', name = 'conv4')(pool3)
    pool4 = MaxPool2D((2 , 2))(conv4)

    conv5 = Conv2D(f1*16 , (3 , 3) , activation='relu' , padding='same', name = 'conv5')(pool4)
    #drop1 = Dropout(0.5)(conv5)

    upsample1 = Conv2D(f1*8, (3 , 3), activation = 'relu', padding = 'same', name = 'upsampling1')(UpSampling2D(size = (2,2))(conv5))
    merged1 = concatenate([conv4, upsample1], name='concatenate1')

    upsample2 = Conv2D(f1*4, (3 , 3), activation = 'relu', padding = 'same', name = 'upsampling2')(UpSampling2D(size = (2,2))(merged1))
    merged2 = concatenate([conv3, upsample2], name='concatenate2')

    upsample3 = Conv2D(f1*2, (3 , 3), activation = 'relu', padding = 'same', name = 'upsampling3')(UpSampling2D(size = (2,2))(merged2))
    merged3 = concatenate([conv2, upsample3], name='concatenate3')

    upsample4 = Conv2D(f1, (3 , 3), activation = 'relu', padding = 'same', name = 'upsampling4')(UpSampling2D(size = (2,2))(merged3))
    merged4 = concatenate([conv1, upsample4], name='concatenate4')

    output = Conv2D(3,(1,1), activation = 'softmax')(merged4)

    return Model(input_img , output)

def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    #bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. We'll need this later to add back to the main path.
    X_shortcut = X
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'same', name = conv_name_base + '2a')(X)
    #X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b')(X)
    #X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'same', name = conv_name_base + '2c')(X)
    #X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet50(input_shape):

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    #X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    conv1 = Conv2D(64, (7, 7), name = 'conv1', padding="same")(X_input)
    #X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    act1 = Activation('relu')(conv1)
    pool1 = MaxPool2D((2 , 2))(act1)

    ident1 = identity_block(pool1, 3, [64, 64, 64], stage=2, block='b')

    conv2 = Conv2D(128, (3, 3), name = 'conv2', padding="same")(ident1)
    act2 = Activation('relu')(conv2)
    pool2 = MaxPool2D((2 , 2))(act2)

    ident2 = identity_block(pool2, 3, [128,128,128], stage=3, block='b')

    conv3 = Conv2D(256, (3, 3), name = 'conv3', padding="same")(ident2)
    act3 = Activation('relu')(conv3)
    pool3 = MaxPool2D((2 , 2))(act3)

    ident3 = identity_block(pool3, 3, [256,256,256], stage=4, block='b')

    conv4 = Conv2D(512, (3, 3), name = 'conv4', padding="same")(ident3)
    act4 = Activation('relu')(conv4)
    pool4 = MaxPool2D((2 , 2))(act4)

    ident4 = identity_block(pool4, 3, [512,512,512], stage=5, block='5')

    conv5 = Conv2D(1024, (3, 3), name = 'conv5', padding="same")(ident4)
    #X  BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    act5 = Activation('relu')(conv5)
    ident5 = identity_block(act5, 3, [1024,1024,1024], stage=6, block='6')


    # Stage 3
    #X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    #X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    #X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    #X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    #X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    #X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')

    # Decoder
    upsampling1 = Conv2D(512, (3 , 3), activation = 'relu', padding = 'same', name = 'upsampling1')(UpSampling2D(size = (2,2))(ident5))
    merged1 = concatenate([conv4, upsampling1], name='concatenate1')

    upsampling2 = Conv2D(256, (3 , 3), activation = 'relu', padding = 'same', name = 'upsampling2')(UpSampling2D(size = (2,2))(merged1))
    merged2 = concatenate([conv3, upsampling2], name='concatenate2')

    upsampling3 = Conv2D(128, (3 , 3), activation = 'relu', padding = 'same', name = 'upsampling3')(UpSampling2D(size = (2,2))(merged2))
    merged3 = concatenate([conv2, upsampling3], name='concatenate3')

    upsampling4 = Conv2D(64, (3 , 3), activation = 'relu', padding = 'same', name = 'upsampling4')(UpSampling2D(size = (2,2))(merged3))
    merged4 = concatenate([conv1, upsampling4], name='concatenate4')

    output = Conv2D(3,(1,1), activation = 'softmax')(merged4)

    model = Model(inputs = X_input, outputs = output, name='ResNet50')

    return model


def test_model(test_x, test_y, model):
    result = model.predict(test_x)
    result1 = result[:,1]
    predicted_class = np.argmax(result, axis=1)
    true_class = test_y
    return predicted_class, true_class, result1

def normalization(image, norm_type = 1):
    image_reshaped = image.reshape((image.shape[0]*image.shape[1]),image.shape[2])
    if (norm_type == 1):
      scaler = StandardScaler()
    if (norm_type == 2):
      scaler = MinMaxScaler(feature_range=(0,1))
    if (norm_type == 3):
      scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(image_reshaped)
    image_normalized = scaler.fit_transform(image_reshaped)
    image_normalized1 = image_normalized.reshape(image.shape[0],image.shape[1],image.shape[2])
    return image_normalized1

def get_patches_batch(image, rows, cols, radio, batch):
    temp = []
    for i in range(0, batch):
        batch_patches = image[rows[i]-radio:rows[i]+radio+1, cols[i]-radio:cols[i]+radio+1, :]
        temp.append(batch_patches)
    patches = np.asarray(temp)
    return patches


def RGB_image(image):
    w, h = image.shape
    RGB= np.zeros((w,h,3)).astype(np.float32)
    for i in range(0,w):
        for j in range(0,h):
            # true negatives
            if image[i,j]==0:
                RGB[i,j,:]=[255,255,255]
            # true positives
            if image[i,j]==1:
                RGB[i,j,:]=[255,255,0]
            # false positives
            if image[i,j]==2:
                RGB[i,j,:]=[255,0,0]
            # false negatives
            if image[i,j]==3:
                RGB[i,j,:]=[0,0,255]
            # past reference
            if image[i,j]==4:
                RGB[i,j,:]=[0,255,0]
    return RGB


def extract_patches(input_image, reference,  patch_size, stride):
    window_shape = patch_size
    # print('debug')
    # print(input_image.shape[2])
    window_shape_array = (window_shape, window_shape, input_image.shape[2])
    window_shape_ref = (window_shape, window_shape)
    patches_array = np.array(view_as_windows(input_image, window_shape_array, step = stride))
    patches_ref = np.array(view_as_windows(reference, window_shape_ref, step = stride))

    num_row,num_col,p,row,col,depth = patches_array.shape
    patches_array = patches_array.reshape(num_row*num_col,row,col,depth)
    patches_ref = patches_ref.reshape(num_row*num_col,row,col)

    return patches_array, patches_ref

def extract_patches_right_region(img_train, img_train_ref, img_mask_ref, patch_size, stride):
    shape = img_train_ref.shape
    patches_train = []
    patches_train_ref = []
    #patches_past_ref = []
    cont_l = 0
    cont_c = 0
    i = 0
    j = 0
    while True:
        if j > shape[1]:
            break
        i = 0
        cont_l = 0
        while True:
            if i > shape[0]:
                break
            patch = img_mask_ref[i:i+patch_size, j:j+patch_size]
            patch_train_ref = img_train_ref[i:i+patch_size, j:j+patch_size]
            patch_train = img_train[i:i+patch_size, j:j+patch_size]
            #patch_past_ref = past_ref[i:i+patch_size, j:j+patch_size]
            # Counts pixels for both classes in the main patch
            unique, counts = np.unique(patch_train_ref, return_counts=True)
            counts_dict = dict(zip(unique, counts))
            # Patch from train reference maybe only contain one of both classes.
            if 1 in counts_dict.keys():
                #print(counts_dict)
                if np.all(patch == -1) == True:
                #if np.all(patch_train_ref != -1) == True:
                    if 0 not in counts_dict.keys():
                        counts_dict[0] = 0
                    total_pixels = counts_dict[0] + counts_dict[1]
                    if counts_dict[1]/total_pixels >= 0.05:
                        patches_train.append(patch_train)
                        patches_train_ref.append(patch_train_ref)
                        #patches_past_ref.append(patch_past_ref)
            i = i + stride
            cont_l +=1
        j = j + stride
        cont_c +=1
    return patches_train, patches_train_ref

def patch_tiles(tiles, mask_amazon, image_array, image_ref, patch_size, stride):
    patches_out = []
    label_out = []
    label_past_out = []
    for num_tile in tiles:
        print('='*60)
        print(num_tile)
        rows, cols = np.where(mask_amazon==num_tile)
        x1 = np.min(rows)
        y1 = np.min(cols)
        x2 = np.max(rows)
        y2 = np.max(cols)

        tile_img = image_array[x1:x2+1,y1:y2+1,:]
        tile_ref = image_ref[x1:x2+1,y1:y2+1]
        # #Alterado
        # unique, counts = np.unique(tile_ref, return_counts=True)
        # counts_dict = dict(zip(unique, counts))
        # print(counts_dict)
        # if 0 not in counts_dict.keys():
        #     counts_dict[0] = 0
        # if 1 not in counts_dict.keys():
        #     counts_dict[1] = 0
        # if 2 not in counts_dict.keys():
        #     counts_dict[2] = 0
        # deforastation = counts_dict[1] / (counts_dict[0] + counts_dict[1] + counts_dict[2])
        # print(f"Deforastation: {deforastation * 100}")
        patches_img, patch_ref = extract_patches(tile_img, tile_ref, patch_size, stride)
        #print(type(patches_img))
        # print(patches_img.shape)
        # print(patch_ref.shape)
        patches_out.append(patches_img)
        label_out.append(patch_ref)

    patches_out = np.concatenate(patches_out)
    label_out = np.concatenate(label_out)
    return patches_out, label_out


def bal_aug_patches(percent, patch_size, patches_img, patches_ref):
    patches_images = []
    patches_labels = []

    for i in range(0,len(patches_img)):
        patch = patches_ref[i]
        class1 = patch[patch==1]

        if len(class1) >= int((patch_size**2)*(percent/100)):
            patch_img = patches_img[i]
            patch_label = patches_ref[i]
            img_aug, label_aug = data_augmentation(patch_img, patch_label)
            patches_images.append(img_aug)
            patches_labels.append(label_aug)

    patches_bal = np.concatenate(patches_images).astype(np.float32)
    labels_bal = np.concatenate(patches_labels).astype(np.float32)
    return patches_bal, labels_bal

def extrac_patch2(img, stride, img_type):
    if img_type == 1:
        h, w = img.shape
        num_patches_h = int(h/stride)
        num_patches_w = int(w/stride)
        patch_t = []
        counter=0
        for i in range(0,num_patches_w):
            for j in range(0,num_patches_h):
                #patch = img[window*i:window*(i+1), window*j:window*(j+1),:]
                patch = img[stride*j:stride*(j+1), stride*i:stride*(i+1)]
                counter=counter+1
                #print(i,j,window*i,window*(i+1), window*j,window*(j+1))
                #print(counter)
                #print(patch.shape)
                patch_t.append(patch)
        patch_t1=np.asarray(patch_t)

    if img_type == 2:
        h, w, c = img.shape
        num_patches_h = int(h/stride)
        num_patches_w = int(w/stride)
        patch_t = []
        counter=0
        for i in range(0,num_patches_w):
            for j in range(0,num_patches_h):
                #patch = img[window*i:window*(i+1), window*j:window*(j+1),:]
                patch = img[stride*j:stride*(j+1), stride*i:stride*(i+1), :]
                counter=counter+1
                #print(i,j,window*i,window*(i+1), window*j,window*(j+1))
                #print(counter)
                #print(patch.shape)
                patch_t.append(patch)
        patch_t1=np.asarray(patch_t)

    return patch_t1

def test_FCN(net, patch_test, patch_test_ref):
    predictions = net.predict(patch_test)
    print(predictions.shape)
    pred1 = predictions[:,:,:,1]

    p_labels=predictions.argmax(axis=3)

    t_vec=np.reshape(patch_test_ref,patch_test_ref.shape[0]*patch_test_ref.shape[1]*patch_test_ref.shape[2])
    p_vec=np.reshape(p_labels,p_labels.shape[0]*p_labels.shape[1]*p_labels.shape[2])
    #prob_vec=np.reshape(pred1,pred1.shape[0]*pred1.shape[1]*pred1.shape[2])
    return p_labels, t_vec, p_vec, pred1

def pred_recostruction(patch_size, pred_labels, image_ref):
    # Reconstruction
    stride = patch_size
    h, w = image_ref.shape
    num_patches_h = int(h/stride)
    num_patches_w = int(w/stride)
    count = 0
    img_reconstructed = np.zeros((num_patches_h*stride,num_patches_w*stride))
    for i in range(0,num_patches_w):
        for j in range(0,num_patches_h):
            img_reconstructed[stride*j:stride*(j+1),stride*i:stride*(i+1)]=pred_labels[count]
            #img_reconstructed[32*i:32*(i+1),32*j:32*(j+1)]=p_labels[count]
            count+=1
    return img_reconstructed

def weighted_categorical_crossentropy(weights):
        """
        A weighted version of keras.objectives.categorical_crossentropy

        Variables:
            weights: numpy array of shape (C,) where C is the number of classes

        Usage:
            weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss,optimizer='adam')
        """

        weights = K.variable(weights)

        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, -1)
            return loss
        return loss

def mask_no_considered(image_ref, buffer, past_ref):
    # Creation of buffer for pixel no considered
    image_ref_ = image_ref.copy()
    im_dilate = skimage.morphology.dilation(image_ref_, disk(buffer))
    outer_buffer = im_dilate-image_ref_
    outer_buffer[outer_buffer==1]=2
    # 1 deforestation, 2 unknown
    final_mask = image_ref_ + outer_buffer
    final_mask[past_ref == 1] = 2
    return final_mask

def prediction(model, image_array, image_ref, final_mask, mask_amazon_ts_, patch_size, area):
    #% Test model
    patch_ts = extrac_patch2(image_array, patch_size, img_type = 2)
    patches_lb = extrac_patch2(image_ref, patch_size, img_type = 1)
    clipping_ref = extrac_patch2(final_mask, patch_size, img_type = 1)

    start_test = time.time()
    p_labels, t_vec, p_vec, probs = test_FCN(model, patch_ts, patches_lb)
    end_test =  time.time() - start_test
    # Reconstruction
    ref_reconstructed = pred_recostruction(patch_size, patches_lb, image_ref)
    img_reconstructed = pred_recostruction(patch_size, p_labels, image_ref)
    prob_recontructed = pred_recostruction(patch_size, probs, image_ref)
    # Não precisava ????
    ref_clip = pred_recostruction(patch_size, clipping_ref, image_ref)

    # ????
    clipping_mask = extrac_patch2(mask_amazon_ts_, patch_size, img_type = 1)
    clipping_mask_ = pred_recostruction(patch_size, clipping_mask, image_ref)

    mask_areas_pred = np.ones_like(ref_reconstructed)
    # O que é isso?
    # Exclui regioes com menos de 69 pixels
    # Sò considera desmatada regioes acima de 69 pixels de desmatamento
    area = skimage.morphology.area_opening(img_reconstructed, area_threshold = area, connectivity=1)
    area_no_consider = img_reconstructed-area
    mask_areas_pred[area_no_consider==1] = 0

    # Mask areas no considered reference (past deforastation)
    mask_borders = np.ones_like(img_reconstructed)
    mask_borders[ref_clip==2] = 0

    # Transforma em 0 tudo que for past deforastation
    # Porque não fazer mask_areas_pred[ref_clip==2] = 0
    mask_no_consider = mask_areas_pred * mask_borders
    ref_consider = mask_no_consider * ref_clip
    pred_consider = mask_no_consider*img_reconstructed

    ref_final = ref_consider[clipping_mask_*mask_no_consider==1]
    pre_final = pred_consider[clipping_mask_*mask_no_consider==1]

    return ref_final, pre_final, prob_recontructed, ref_reconstructed, ref_clip, clipping_mask_, end_test


def color_map(prob_map, ref_reconstructed, mask_no_considered, clipping_mask_, th):
    reconstructed = prob_map.copy()
    reconstructed[reconstructed >= th] = 1
    reconstructed[reconstructed < th] = 0

    true_positives = (reconstructed*ref_reconstructed)
    diff_image = reconstructed-ref_reconstructed

    output_map = np.zeros((ref_reconstructed.shape)).astype(np.float32)
    output_map[true_positives == 1] = 1
    output_map[diff_image == 1] = 2
    output_map[diff_image==-1] = 3
    output_map[mask_no_considered == 2] = 4
    output_map[clipping_mask_ == 0] = 0
    return output_map
