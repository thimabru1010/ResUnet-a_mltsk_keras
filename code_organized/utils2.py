import numpy as np
from utils import data_augmentation
import time

def extract_patches_right_region(img_train, img_train_ref, img_mask_ref, patch_size, stride, percent):
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
                # if np.all(patch_train_ref != -1) == True and patch_train_ref.shape == (patch_size, patch_size):
                    if 0 not in counts_dict.keys():
                        counts_dict[0] = 0
                    total_pixels = counts_dict[0] + counts_dict[1]
                    #print(counts_dict[1]/total_pixels)
                    if counts_dict[1]/total_pixels >= percent/100:
                        patches_train.append(np.asarray(patch_train))
                        patches_train_ref.append(np.asarray(patch_train_ref))
                        #patches_past_ref.append(patch_past_ref)
            i = i + stride
            cont_l +=1
        j = j + stride
        cont_c +=1
    return patches_train, patches_train_ref

def extract_patches_right_region_prediction(img_train, img_train_ref, mask_amazon_ts_, final_mask, patch_size, stride):
    shape = img_train_ref.shape
    patches_train = []
    patches_train_ref = []
    patches_past_ref = []
    patches_mask_amazon = []
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
            patch_mask_amazon = mask_amazon_ts_[i:i+patch_size, j:j+patch_size]
            patch_train_ref = img_train_ref[i:i+patch_size, j:j+patch_size]
            patch_train = img_train[i:i+patch_size, j:j+patch_size]
            #patch_past_ref = final_mask[i:i+patch_size, j:j+patch_size]
            if np.all(patch_train_ref != -1) == True and patch_train_ref.shape == (patch_size, patch_size):
                # if 0 not in counts_dict.keys():
                #     counts_dict[0] = 0
                # total_pixels = counts_dict[0] + counts_dict[1]
                #if counts_dict[1]/total_pixels >= 0.05:
                patches_train.append(np.asarray(patch_train))
                patches_train_ref.append(np.asarray(patch_train_ref))
                #patches_past_ref.append(patch_past_ref)
                #patches_mask_amazon.append(patch_mask_amazon)
            i = i + stride
            cont_l +=1
        j = j + stride
        cont_c +=1
    return patches_train, patches_train_ref, patches_past_ref, patches_mask_amazon

def patch_tiles_prediction(tiles, mask_amazon, image_array, image_ref, img_mask_ref, patch_size, stride):
    patches_out = []
    label_out = []
    label_past_out = []
    for num_tile in tiles:
        print(f"Num tile: {num_tile}")
        rows, cols = np.where(mask_amazon==1)
        print(rows, cols)
        x1 = np.min(rows)
        y1 = np.min(cols)
        x2 = np.max(rows)
        y2 = np.max(cols)

        tile_img = image_array[x1:x2+1,y1:y2+1,:]
        tile_ref = image_ref[x1:x2+1,y1:y2+1]
        # patches_img, patch_ref = extract_patches(tile_img, tile_ref, patch_size, stride)
        patches_img = patches_with_out_overlap(tile_img, stride, 2, tile_ref)
        patch_ref = patches_with_out_overlap(tile_ref, stride, 1, tile_ref)

        # patches_img, patch_ref, _, _ = extract_patches_right_region_prediction(tile_img, tile_ref, mask_amazon, img_mask_ref, patch_size, stride)

        #if len(patch_ref) > 0:
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


def patch_tiles2(tiles, mask_amazon, image_array, image_ref, img_mask_ref, patch_size, stride, percent):
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
        tile_mask_ref = img_mask_ref[x1:x2+1,y1:y2+1]
        # patches_img, patch_ref = extract_patches(tile_img, tile_ref, patch_size, stride)
        patches_img, patch_ref = extract_patches_right_region(tile_img, tile_ref, tile_mask_ref, patch_size, stride, percent)

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

def test_FCN2(net, patch_test):
    ''' Function to test FCN model'''
    predictions = net.predict(patch_test)

    print(predictions.shape)
    pred1 = predictions[:,:,:,1]
    p_labels=predictions.argmax(axis=-1)
    return p_labels, pred1

def patches_with_out_overlap(img, stride, img_type, img_ref=None):
    '''Extract patches without overlap to test models, img_type = 1 (reference image), img_type = 2 (images)'''
    patch_size = stride
    if img_type == 1:
        h, w = img.shape
        num_patches_h = int(h/stride)
        num_patches_w = int(w/stride)
        patch_t = []
        counter=0
        for i in range(0,num_patches_w):
            for j in range(0,num_patches_h):
                patch = img[stride*j:stride*(j+1), stride*i:stride*(i+1)]
                patch_ref = img_ref[stride*j:stride*(j+1), stride*i:stride*(i+1)]
                if np.all(patch_ref != -1) == True and patch_ref.shape == (patch_size, patch_size):
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
                patch_ref = img_ref[stride*j:stride*(j+1), stride*i:stride*(i+1)]
                if np.all(patch_ref != -1) == True and patch_ref.shape == (patch_size, patch_size):
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
    p_labels, probs = test_FCN2(model, patch_ts)
    end_test =  time.time() - start_test
    prob_recontructed = pred_recostruction(patch_size, probs, final_mask)
    return prob_recontructed, end_test

def matrics_AA_recall(thresholds, prob_map, reference, mask_amazon_ts, area):
    metrics_all = []

    for thr in thresholds:
        print(thr)
        img_reconstructed = prob_map.copy()
        img_reconstructed[img_reconstructed >= thr] = 1
        img_reconstructed[img_reconstructed < thr] = 0
        #plt.imshow(img_reconstructed)

        mask_areas_pred = np.ones_like(reference)
        area = skimage.morphology.area_opening(img_reconstructed, area_threshold=area, connectivity=1)
        area_no_consider = img_reconstructed-area
        mask_areas_pred[area_no_consider==1] = 0

        # Mask areas no considered reference
        mask_borders = np.ones_like(img_reconstructed)
        mask_borders[reference==2] = 0

        mask_no_consider = mask_areas_pred * mask_borders
        ref_consider = mask_no_consider * reference
        pred_consider = mask_no_consider*img_reconstructed

        ref_final = ref_consider[mask_amazon_ts==1]
        pre_final = pred_consider[mask_amazon_ts==1]

        # Metrics
        cm = confusion_matrix(ref_final, pre_final)
        metrics = compute_metrics(ref_final, pre_final)
        print('Confusion  matrix \n', cm)
        print('Accuracy: ', metrics[0])
        print('F1score: ', metrics[1])
        print('Recall: ', metrics[2])
        print('Precision: ', metrics[3])
        #TN = cm[0,0]
        FN = cm[1,0]
        TP = cm[1,1]
        FP = cm[0,1]
        precision_ = TP/(TP+FP)
        recall_ = TP/(TP+FN)
        aa = (TP+FP)/len(ref_final)
        mm = np.hstack((recall_, precision_, aa))
        metrics_all.append(mm)
    metrics_ = np.asarray(metrics_all)
    return metrics_

def test_FCN(net, patch_test, patch_test_ref):
    predictions = net.predict(patch_test)
    print(predictions.shape)
    pred1 = predictions[:,:,:,1]

    p_labels=predictions.argmax(axis=3)

    t_vec=np.reshape(patch_test_ref,patch_test_ref.shape[0]*patch_test_ref.shape[1]*patch_test_ref.shape[2])
    p_vec=np.reshape(p_labels,p_labels.shape[0]*p_labels.shape[1]*p_labels.shape[2])
    #prob_vec=np.reshape(pred1,pred1.shape[0]*pred1.shape[1]*pred1.shape[2])
    return p_labels, t_vec, p_vec, pred1

def prediction2(model, image_array, image_ref, final_mask, mask_amazon_ts_, patch_size, area):
    #% Test model
    # patch_ts = extrac_patch2(image_array, patch_size, img_type = 2)
    # patches_lb = extrac_patch2(image_ref, patch_size, img_type = 1)
    # clipping_ref = extrac_patch2(final_mask, patch_size, img_type = 1)

    patch_ts, patches_lb, clipping_ref, clipping_mask = extract_patches_right_region_prediction(image_array, image_ref, mask_amazon_ts_, final_mask, patch_size, stride=patch_size)
    print('extraiu...')

    patch_ts, patches_lb, clipping_ref, clipping_mask = np.asarray(patch_ts), np.asarray(patches_lb), np.asarray(clipping_ref), np.asarray(clipping_mask)


    start_test = time.time()
    p_labels, _, _, probs = test_FCN(model, patch_ts, patches_lb)
    end_test =  time.time() - start_test
    # Reconstruction
    ref_reconstructed = pred_recostruction(patch_size, patches_lb, image_ref)
    img_reconstructed = pred_recostruction(patch_size, p_labels, image_ref)
    prob_recontructed = pred_recostruction(patch_size, probs, image_ref)
    # Não precisava ????
    ref_clip = pred_recostruction(patch_size, clipping_ref, image_ref)

    # ????
    #clipping_mask = extrac_patch2(mask_amazon_ts_, patch_size, img_type = 1)
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
