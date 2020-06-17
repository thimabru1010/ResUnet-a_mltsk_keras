import numpy as np
from utils import data_augmentation

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
                if np.all(patch == -1) == True and patch_train_ref.shape == (patch_size, patch_size):
                #if np.all(patch_train_ref != -1) == True:
                    if 0 not in counts_dict.keys():
                        counts_dict[0] = 0
                    total_pixels = counts_dict[0] + counts_dict[1]
                    if counts_dict[1]/total_pixels >= 0.05:
                        patches_train.append(np.asarray(patch_train))
                        patches_train_ref.append(np.asarray(patch_train_ref))
                        #patches_past_ref.append(patch_past_ref)
            i = i + stride
            cont_l +=1
        j = j + stride
        cont_c +=1
    return patches_train, patches_train_ref


def patch_tiles2(tiles, mask_amazon, image_array, image_ref, img_mask_ref, patch_size, stride):
    patches_out = []
    label_out = []
    label_past_out = []
    for num_tile in tiles:
        rows, cols = np.where(mask_amazon==num_tile)
        x1 = np.min(rows)
        y1 = np.min(cols)
        x2 = np.max(rows)
        y2 = np.max(cols)

        tile_img = image_array[x1:x2+1,y1:y2+1,:]
        tile_ref = image_ref[x1:x2+1,y1:y2+1]
        # patches_img, patch_ref = extract_patches(tile_img, tile_ref, patch_size, stride)
        patches_img, patch_ref = extract_patches_right_region(tile_img, tile_ref, img_mask_ref, patch_size, stride)

        if len(patch_ref) > 0:
            patches_out.append(np.asarray(patches_img))
            label_out.append(np.asarray(patch_ref))

        print('patches tudo')
        print(len(patches_img))
        print(len(patch_ref))

    print('tudo')
    print(len(patches_out))
    print(len(label_out))
    patches_out = np.concatenate(patches_out)
    label_out = np.concatenate(label_out)
    print(patches_out.shape)
    print(label_out.shape)
    return patches_out, label_out

def patch_tiles3(tiles, mask_amazon, image_array, image_ref, patch_size, stride):
    patches_out = []
    label_out = []
    label_past_out = []
    unique, counts = np.unique(image_ref, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    print(counts_dict)
    total_pixels_ref = counts_dict[0] + counts_dict[1] + counts_dict[2]
    total_def = counts_dict[1]
    tile_def_dict = {}
    for num_tile in tiles:
        print('='*60)
        print(f"Patch tile number: {num_tile}")
        rows, cols = np.where(mask_amazon==num_tile)
        x1 = np.min(rows)
        y1 = np.min(cols)
        x2 = np.max(rows)
        y2 = np.max(cols)

        tile_img = image_array[x1:x2+1,y1:y2+1,:]
        tile_ref = image_ref[x1:x2+1,y1:y2+1]
        #Alterado
        unique, counts = np.unique(tile_ref, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        print(counts_dict)
        if 0 not in counts_dict.keys():
            counts_dict[0] = 0
        if 1 not in counts_dict.keys():
            counts_dict[1] = 0
        if 2 not in counts_dict.keys():
            counts_dict[2] = 0
        deforastation_image = counts_dict[1] / total_pixels_ref
        deforastation_only = counts_dict[1] / total_def
        #print(f"Deforastation %: {deforastation_image * 100}")
        print(f"Deforastation only %: {deforastation_only * 100:.3f}")
        tile_def_dict[num_tile] = round(deforastation_only*100, 3)
    #     patches_img, patch_ref = extract_patches(tile_img, tile_ref, patch_size, stride)
    #     #print(type(patches_img))
    #     # print(patches_img.shape)
    #     # print(patch_ref.shape)
    #     patches_out.append(patches_img)
    #     label_out.append(patch_ref)
    #
    # patches_out = np.concatenate(patches_out)
    # label_out = np.concatenate(label_out)
    print('='*60)
    print(tile_def_dict)
    return patches_out, label_out

def bal_aug_patches2(percent, patch_size, patches_img, patches_ref):
    patches_images = []
    patches_labels = []
    print('bal_aug_patches')

    print(len(patches_img))
    print(patches_img.shape)
    for i in range(0,len(patches_img)):
        patch = patches_ref[i]
        class1 = patch[patch==1]

        #print('class1')
        # print(len(class1))
        # print(int((patch_size**2)*(percent/100)))
        #if len(class1) >= int((patch_size**2)*(percent/100)):
        patch_img = patches_img[i]
        patch_label = patches_ref[i]
        # print(patch_label.shape)
        # print(patch_img.shape)
        img_aug, label_aug = data_augmentation(patch_img, patch_label)
        # print(img_aug.shape)
        # print(label_aug.shape)
        patches_images.append(img_aug)
        patches_labels.append(label_aug)

    print(len(patches_images))
    patches_bal = np.concatenate(patches_images).astype(np.float32)
    labels_bal = np.concatenate(patches_labels).astype(np.float32)
    return patches_bal, labels_bal

def bal_aug_patches3(percent, patch_size, patches_img, patches_ref):
    patches_images = []
    patches_labels = []
    print('bal_aug_patches')

    print(len(patches_img))
    for i in range(0,len(patches_img)):
        patch = patches_ref[i]
        class1 = patch[patch==1]

        # o que é esse cálculo
        #print('class1')
        # print(len(class1))
        # print(int((patch_size**2)*(percent/100)))
        if len(class1) >= int((patch_size**2)*(percent/100)) and np.all(patch != -1) == True:
            patch_img = patches_img[i]
            patch_label = patches_ref[i]
            img_aug, label_aug = data_augmentation(patch_img, patch_label)
            patches_images.append(img_aug)
            patches_labels.append(label_aug)

    print(len(patches_images))
    patches_bal = np.concatenate(patches_images).astype(np.float32)
    labels_bal = np.concatenate(patches_labels).astype(np.float32)
    return patches_bal, labels_bal

def test_FCN(net, patch_test):
    ''' Function to test FCN model'''
    predictions = net.predict(patch_test)
    print(predictions.shape)
    pred1 = predictions[:,:,:,1]
    p_labels=predictions.argmax(axis=-1)
    return p_labels, pred1

def patches_with_out_overlap(img, stride, img_type):
    '''Extract patches without overlap to test models, img_type = 1 (reference image), img_type = 2 (images)'''
    if img_type == 1:
        h, w = img.shape
        num_patches_h = int(h/stride)
        num_patches_w = int(w/stride)
        patch_t = []
        counter=0
        for i in range(0,num_patches_w):
            for j in range(0,num_patches_h):
                patch = img[stride*j:stride*(j+1), stride*i:stride*(i+1)]
                counter=counter+1
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
                patch = img[stride*j:stride*(j+1), stride*i:stride*(i+1), :]
                counter=counter+1
                patch_t.append(patch)
        patch_t1=np.asarray(patch_t)

    return patch_t1

def pred_recostruction(patch_size, pred_labels, image_ref):
    ''' Reconstruction of whole prediction image'''
    stride = patch_size
    h, w = image_ref.shape
    num_patches_h = int(h/stride)
    num_patches_w = int(w/stride)
    count = 0
    img_reconstructed = np.zeros((num_patches_h*stride,num_patches_w*stride))
    for i in range(0,num_patches_w):
        for j in range(0,num_patches_h):
            img_reconstructed[stride*j:stride*(j+1),stride*i:stride*(i+1)]=pred_labels[count]
            count+=1
    return img_reconstructed

def output_prediction_FC(model, image_array, final_mask, patch_size):
    start_test = time.time()
    patch_ts = patches_with_out_overlap(image_array, patch_size, img_type = 2)
    p_labels, probs = test_FCN(model, patch_ts)
    end_test =  time.time() - start_test
    prob_recontructed = pred_recostruction(patch_size, probs, final_mask)
    return prob_recontructed, end_test
