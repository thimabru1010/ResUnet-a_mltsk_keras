import numpy as np
import tensorflow.keras
#from utils import data_augmentation
from tensorflow.keras.utils import Sequence

def data_augmentation(image, labels):
    aug_imgs = np.zeros((5, image.shape[0], image.shape[1], image.shape[2]))
    aug_lbs = np.zeros((5, labels.shape[0], labels.shape[1], labels.shape[2]))

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
        aug_lbs[0, :, :, :] = labels
        aug_lbs[1, :, :, :] = np.rot90(labels, 1)
        aug_lbs[2, :, :, :] = np.rot90(labels, 2)
        #aug_lbs[3, :, :] = np.rot90(labels, 3)
        #horizontal_flip_lb = np.flip(labels,0)
        aug_lbs[3, :, :, :] = np.flip(labels,0)
        aug_lbs[4, :, :, :] = np.flip(labels, 1)
        #aug_lbs[6, :, :] = np.rot90(horizontal_flip_lb, 2)
        #aug_lbs[7, :, :] =np.rot90(horizontal_flip_lb, 3)

    return aug_imgs, aug_lbs

class Mygenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Initialise X_train and y_train arrays for this batch
        X_train = []
        y_train = []

        # For each example
        #for batch_sample in batch_samples:
        for i in range(len(batch_x)):
            # Load image (X) and label (y)
            img = batch_x[i, :, :, :]
            label = batch_y[i, :, :, :]

            # apply any kind of preprocessing
            img, label = data_augmentation(img, label)
            #img = cv2.resize(img,(resize,resize))

            # Add example to arrays
            X_train.append(img)
            y_train.append(label)

        # Make sure they're numpy arrays (as opposed to lists)
        X_train = np.concatenate(X_train).astype(np.float32)
        y_train = np.concatenate(y_train).astype(np.float32)

        # read your data here using the batch lists, batch_x and batch_y
        # x = [my_readfunction(filename) for filename in batch_x]
        # y = [my_readfunction(filename) for filename in batch_y]
        return X_train, y_train
