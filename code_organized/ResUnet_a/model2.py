import tensorflow.keras.models as KM
import tensorflow.keras as KE
import tensorflow.keras.layers as KL
#import tensorflow.keras.engine as KE
import tensorflow.keras.backend as KB
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf

from ResUnet_a.config import UnetConfig
import utils
import math
import numpy as np

print('[TFMERDA1]'*10)
tf.compat.v1.enable_eager_execution()
print( tf.executing_eagerly() )

class Resunet_a2(object):
    def __init__(self, input_shape, num_classes, config=UnetConfig()):
        self.num_classes = num_classes
        self.config = config
        print(f"Input shape: {input_shape}")
        self.img_height, self.img_width, self.img_channel = input_shape
        self.model = self.build_model_ResUneta()

    def build_model_ResUneta(self):
        def Tanimoto_loss(label,pred):
            square=tf.square(pred)
            sum_square=tf.reduce_sum(square,axis=-1)
            product=tf.multiply(pred,label)
            sum_product=tf.reduce_sum(product,axis=-1)
            denomintor=tf.subtract(tf.add(sum_square,1),sum_product)
            loss=tf.divide(sum_product,denomintor)
            loss=tf.reduce_mean(loss)
            return 1.0-loss

        def Tanimoto_dual_loss(label,pred):
            loss1=Tanimoto_loss(pred,label)
            pred=tf.subtract(1.0,pred)
            label=tf.subtract(1.0,label)
            loss2=Tanimoto_loss(label,pred)
            loss=(loss1+loss2)/2

        def ResBlock(input,filter,kernel_size,dilation_rates,stride):
            def branch(dilation_rate):
                x=KL.BatchNormalization()(input)
                x=KL.Activation('relu')(x)
                x=KL.Conv2D(filter,kernel_size,strides=stride,dilation_rate=dilation_rate,padding='same')(x)
                x=KL.BatchNormalization()(x)
                x=KL.Activation('relu')(x)
                x=KL.Conv2D(filter,kernel_size,strides=stride,dilation_rate=dilation_rate,padding='same')(x)
                return x
            out=[]
            for d in dilation_rates:
                out.append(branch(d))
            if len(dilation_rates)>1:
                out=KL.Add()(out)
            else:
                out=out[0]
            return out
        def PSPPooling(input,filter):
            print('[DEBUG]'*10)
            print(input.shape)
            print('[DEBUG]'*10)
            x1=KL.MaxPooling2D(pool_size=(1,1), padding='same')(input)
            x2=KL.MaxPooling2D(pool_size=(2,2), padding='same')(input)
            x3=KL.MaxPooling2D(pool_size=(4,4), padding='same')(input)
            x4=KL.MaxPooling2D(pool_size=(8,8), padding='same')(input)
            x1=KL.Conv2D(int(filter/4),(1,1), padding='same')(x1)
            x2=KL.Conv2D(int(filter/4),(1,1), padding='same')(x2)
            x3=KL.Conv2D(int(filter/4),(1,1), padding='same')(x3)
            x4=KL.Conv2D(int(filter/4),(1,1), padding='same')(x4)
            n_input = input.shape[1]
            print(x1.shape)
            up_size_x1 = (n_input // x1.shape[1], n_input // x1.shape[2])
            print(x2.shape)
            up_size_x2 = (n_input // x2.shape[1], n_input // x2.shape[2])
            print(x3.shape)
            up_size_x3 = (n_input // x3.shape[1], n_input // x3.shape[2])
            print(x4.shape)
            up_size_x4 = (n_input // x4.shape[1], n_input // x4.shape[2])
            # x1=KL.UpSampling2D(size=(2,2))(x1)
            # x2=KL.UpSampling2D(size=(4,4))(x2)
            # x3=KL.UpSampling2D(size=(8,8))(x3)
            # x4=KL.UpSampling2D(size=(16,16))(x4)
            print('Upsamples sizes:')
            print(up_size_x1)
            print(up_size_x2)
            print(up_size_x3)
            print(up_size_x4)
            x1=KL.UpSampling2D(size=up_size_x1)(x1)
            x2=KL.UpSampling2D(size=up_size_x2)(x2)
            x3=KL.UpSampling2D(size=up_size_x3)(x3)
            x4=KL.UpSampling2D(size=up_size_x4)(x4)
            print(x1.shape)
            print(x2.shape)
            print(x3.shape)
            print(x4.shape)
            x=KL.Concatenate()([x1,x2,x3,x4,input])
            x=KL.Conv2D(filter,(1,1))(x)
            return x

        def combine(input1,input2,filter):
            x=KL.Activation('relu')(input1)
            x=KL.Concatenate()([x,input2])
            x=KL.Conv2D(filter,(1,1))(x)
            return x
        # inputs=KM.Input(shape=(self.config.IMAGE_H, self.config.IMAGE_W, self.config.IMAGE_C))
        #KB.set_image_dim_ordering('th')
        inputs=KE.Input(shape=(self.img_height, self.img_width, self.img_channel))

        # Encoder
        c1=x=KL.Conv2D(32,(1,1),strides=(1,1),dilation_rate=1, padding='same')(inputs)
        print(x.shape)
        c2=x=ResBlock(x,32,(3,3),[1,3,15,31],(1,1))
        print(x.shape)

        x=KL.Conv2D(64,(1,1),strides=(2,2), padding='same')(x)
        c3=x=ResBlock(x,64,(3,3),[1,3,15,31],(1,1))
        print(x.shape)

        x=KL.Conv2D(128,(1,1),strides=(2,2), padding='same')(x)
        c4=x=ResBlock(x,128,(3,3),[1,3,15],(1,1))
        print(x.shape)
        print('aqui'*20)

        x=KL.Conv2D(256,(1,1),strides=(2,2), padding='same')(x)
        c5=x=ResBlock(x,256,(3,3),[1,3,15],(1,1))
        print(x.shape)

        x=KL.Conv2D(512,(1,1),strides=(2,2), padding='same')(x)
        c6=x=ResBlock(x,512,(3,3),[1],(1,1))
        print(x.shape)

        # Talvez isso deva ser sempre 1024
        x=KL.Conv2D(1024,(1,1),strides=(2,2), padding='same')(x)
        x=ResBlock(x,1024,(3,3),[1],(1,1))

        print('[DEBUG]'*10)
        print(x.shape)

        x=PSPPooling(x,1024)

        # Decoder
        x=KL.Conv2D(512,(1,1))(x)
        x=KL.UpSampling2D()(x)
        x=combine(x,c6,512)
        x=ResBlock(x,512,(3,3),[1],1)

        x=KL.Conv2D(256,(1,1))(x)
        x=KL.UpSampling2D()(x)
        x=combine(x,c5,256)
        x=ResBlock(x,256,(3,3),[1,3,15],1)

        x=KL.Conv2D(128,(1,1))(x)
        x=KL.UpSampling2D()(x)
        x=combine(x,c4,128)
        x=ResBlock(x,128,(3,3),[1,3,15],1)

        x=KL.Conv2D(64,(1,1))(x)
        x=KL.UpSampling2D()(x)
        x=combine(x,c3,64)
        x=ResBlock(x,64,(3,3),[1,3,15,31],1)

        x=KL.Conv2D(32,(1,1))(x)
        x=KL.UpSampling2D()(x)
        x=combine(x,c2,32)

        x=ResBlock(x,32,(3,3),[1,3,15,31],1)
        x=combine(x,c1,32)

        x=PSPPooling(x,32)
        x=KL.Conv2D(self.num_classes,(1,1))(x)
        x=KL.Activation('softmax')(x)
        model=KM.Model(inputs=inputs,outputs=x)

        return model

    def train(self, data_path, model_file, restore_model_file=None):
        model = self.model
        if restore_model_file:
            model.load_weights(restore_model_file)
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=model_file,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(model_file+"/Unet{epoch:02d}.h5",
                                            verbose=0, save_weights_only=True),
        ]


        train_datasets=utils.DataGenerator_wqw(data_path+"/train/image",data_path+"/train/label",self.config.IMAGE_H,self.config.IMAGE_H,self.config.batch_size,self.config.CLASSES_NUM,self.config)
        val_datasets=utils.DataGenerator_wqw(data_path+"/val/image",data_path+"/val/label",self.config.IMAGE_H,self.config.IMAGE_H,self.config.batch_size,self.config.CLASSES_NUM,self.config)

        print ("the number of train data is", len(train_datasets))
        print ("the number of val data is", len(val_datasets))
        trainCounts = len(train_datasets)
        valCounts = len(val_datasets)
        model.fit_generator(generator=train_datasets,epochs=self.config.EPOCHS,validation_data=val_datasets,callbacks=callbacks, max_queue_size=10,workers=8,use_multiprocessing=True)


class Resunet_a2_multitasking(object):
    def __init__(self, input_shape, num_classes, config=UnetConfig()):
        self.num_classes = num_classes
        self.config = config
        print(f"Input shape: {input_shape}")
        self.img_height, self.img_width, self.img_channel = input_shape
        self.model = self.build_model_ResUneta()

    def build_model_ResUneta(self):
        def Tanimoto_loss(label,pred):
            square=tf.square(pred)
            sum_square=tf.reduce_sum(square,axis=-1)
            product=tf.multiply(pred,label)
            sum_product=tf.reduce_sum(product,axis=-1)
            denomintor=tf.subtract(tf.add(sum_square,1),sum_product)
            loss=tf.divide(sum_product,denomintor)
            loss=tf.reduce_mean(loss)
            return 1.0-loss

        def Tanimoto_dual_loss(label,pred):
            loss1=Tanimoto_loss(pred,label)
            pred=tf.subtract(1.0,pred)
            label=tf.subtract(1.0,label)
            loss2=Tanimoto_loss(label,pred)
            loss=(loss1+loss2)/2

        def ResBlock(input,filter,kernel_size,dilation_rates,stride):
            def branch(dilation_rate):
                x=KL.BatchNormalization()(input)
                x=KL.Activation('relu')(x)
                x=KL.Conv2D(filter,kernel_size,strides=stride,dilation_rate=dilation_rate,padding='same')(x)
                x=KL.BatchNormalization()(x)
                x=KL.Activation('relu')(x)
                x=KL.Conv2D(filter,kernel_size,strides=stride,dilation_rate=dilation_rate,padding='same')(x)
                return x
            out=[]
            for d in dilation_rates:
                out.append(branch(d))
            if len(dilation_rates)>1:
                out=KL.Add()(out)
            else:
                out=out[0]
            return out
        def PSPPooling(input,filter):
            print('[DEBUG]'*10)
            print(input.shape)
            print('[DEBUG]'*10)
            x1=KL.MaxPooling2D(pool_size=(1,1), padding='same')(input)
            x2=KL.MaxPooling2D(pool_size=(2,2), padding='same')(input)
            x3=KL.MaxPooling2D(pool_size=(4,4), padding='same')(input)
            x4=KL.MaxPooling2D(pool_size=(8,8), padding='same')(input)
            x1=KL.Conv2D(int(filter/4),(1,1), padding='same')(x1)
            x2=KL.Conv2D(int(filter/4),(1,1), padding='same')(x2)
            x3=KL.Conv2D(int(filter/4),(1,1), padding='same')(x3)
            x4=KL.Conv2D(int(filter/4),(1,1), padding='same')(x4)
            n_input = input.shape[1]
            print(x1.shape)
            up_size_x1 = (n_input // x1.shape[1], n_input // x1.shape[2])
            print(x2.shape)
            up_size_x2 = (n_input // x2.shape[1], n_input // x2.shape[2])
            print(x3.shape)
            up_size_x3 = (n_input // x3.shape[1], n_input // x3.shape[2])
            print(x4.shape)
            up_size_x4 = (n_input // x4.shape[1], n_input // x4.shape[2])
            # x1=KL.UpSampling2D(size=(2,2))(x1)
            # x2=KL.UpSampling2D(size=(4,4))(x2)
            # x3=KL.UpSampling2D(size=(8,8))(x3)
            # x4=KL.UpSampling2D(size=(16,16))(x4)
            print('Upsamples sizes:')
            print(up_size_x1)
            print(up_size_x2)
            print(up_size_x3)
            print(up_size_x4)
            x1=KL.UpSampling2D(size=up_size_x1)(x1)
            x2=KL.UpSampling2D(size=up_size_x2)(x2)
            x3=KL.UpSampling2D(size=up_size_x3)(x3)
            x4=KL.UpSampling2D(size=up_size_x4)(x4)
            print(x1.shape)
            print(x2.shape)
            print(x3.shape)
            print(x4.shape)
            x=KL.Concatenate()([x1,x2,x3,x4,input])
            x=KL.Conv2D(filter,(1,1))(x)
            return x

        def combine(input1,input2,filter):
            x=KL.Activation('relu')(input1)
            x=KL.Concatenate()([x,input2])
            x=KL.Conv2D(filter,(1,1))(x)
            return x
        # inputs=KM.Input(shape=(self.config.IMAGE_H, self.config.IMAGE_W, self.config.IMAGE_C))
        #KB.set_image_dim_ordering('th')
        inputs=KE.Input(shape=(self.img_height, self.img_width, self.img_channel))

        # Encoder
        c1=x=KL.Conv2D(32,(1,1),strides=(1,1),dilation_rate=1, padding='same')(inputs)
        print(x.shape)
        c2=x=ResBlock(x,32,(3,3),[1,3,15,31],(1,1))
        print(x.shape)

        x=KL.Conv2D(64,(1,1),strides=(2,2), padding='same')(x)
        c3=x=ResBlock(x,64,(3,3),[1,3,15,31],(1,1))
        print(x.shape)

        x=KL.Conv2D(128,(1,1),strides=(2,2), padding='same')(x)
        c4=x=ResBlock(x,128,(3,3),[1,3,15],(1,1))
        print(x.shape)
        print('aqui'*20)

        x=KL.Conv2D(256,(1,1),strides=(2,2), padding='same')(x)
        c5=x=ResBlock(x,256,(3,3),[1,3,15],(1,1))
        print(x.shape)

        x=KL.Conv2D(512,(1,1),strides=(2,2), padding='same')(x)
        c6=x=ResBlock(x,512,(3,3),[1],(1,1))
        print(x.shape)

        # Talvez isso deva ser sempre 1024
        x=KL.Conv2D(1024,(1,1),strides=(2,2), padding='same')(x)
        x=ResBlock(x,1024,(3,3),[1],(1,1))

        print('[DEBUG]'*10)
        print(x.shape)

        x=PSPPooling(x,1024)

        # Decoder
        x=KL.Conv2D(512,(1,1))(x)
        x=KL.UpSampling2D()(x)
        x=combine(x,c6,512)
        x=ResBlock(x,512,(3,3),[1],1)

        x=KL.Conv2D(256,(1,1))(x)
        x=KL.UpSampling2D()(x)
        x=combine(x,c5,256)
        x=ResBlock(x,256,(3,3),[1,3,15],1)

        x=KL.Conv2D(128,(1,1))(x)
        x=KL.UpSampling2D()(x)
        x=combine(x,c4,128)
        x=ResBlock(x,128,(3,3),[1,3,15],1)

        x=KL.Conv2D(64,(1,1))(x)
        x=KL.UpSampling2D()(x)
        x=combine(x,c3,64)
        x=ResBlock(x,64,(3,3),[1,3,15,31],1)

        x=KL.Conv2D(32,(1,1))(x)
        x=KL.UpSampling2D()(x)
        x=combine(x,c2,32)

        x=ResBlock(x,32,(3,3),[1,3,15,31],1)
        x_comb=combine(x,c1,32)

        x_psp=PSPPooling(x_comb,32)
        print('PSPPooling')
        print(x_psp.shape)

        # Segmentation
        # OBS para o jeito de inserir o padding

        x_seg=KL.Conv2D(32,(3,3), activation='relu', padding='same', name='seg1')(x_psp)
        x_seg=KL.Conv2D(32,(3,3), activation='relu', padding='same', name='seg2')(x_seg)
        x_seg=KL.Conv2D(self.num_classes,(1,1), padding='valid', name='seg3')(x_seg)
        out_seg=KL.Activation('softmax', name='segmentation')(x_seg)

        # Boundary
        x_bound=KL.Conv2D(32,(3,3), activation='relu', padding='same')(x_psp)
        x_bound=KL.Conv2D(1,(1,1), padding='valid')(x_bound)
        out_bound=KL.Activation('sigmoid', name='boundary')(x_bound)

        # Distance
        x_dist=KL.Conv2D(32,(3,3), activation='relu', padding='same')(x_comb)
        x_dist=KL.Conv2D(32,(3,3), activation='relu', padding='same')(x_dist)
        x_dist=KL.Conv2D(1,(1,1), padding='valid')(x_dist)
        out_dist=KL.Activation('softmax', name='distance')(x_dist)

        # Color
        # Talvez mudar para same
        out_color=KL.Conv2D(3,(1,1), activation='sigmoid', padding='valid', name='color')(x_comb)

        out = [out_seg, out_bound, out_dist, out_color]

        model=KM.Model(inputs=inputs,outputs=out)
        return model

    def train(self, data_path, model_file, restore_model_file=None):
        model = self.model
        if restore_model_file:
            model.load_weights(restore_model_file)
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=model_file,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(model_file+"/Unet{epoch:02d}.h5",
                                            verbose=0, save_weights_only=True),
        ]


        train_datasets=utils.DataGenerator_wqw(data_path+"/train/image",data_path+"/train/label",self.config.IMAGE_H,self.config.IMAGE_H,self.config.batch_size,self.config.CLASSES_NUM,self.config)
        val_datasets=utils.DataGenerator_wqw(data_path+"/val/image",data_path+"/val/label",self.config.IMAGE_H,self.config.IMAGE_H,self.config.batch_size,self.config.CLASSES_NUM,self.config)

        print ("the number of train data is", len(train_datasets))
        print ("the number of val data is", len(val_datasets))
        trainCounts = len(train_datasets)
        valCounts = len(val_datasets)
        model.fit_generator(generator=train_datasets,epochs=self.config.EPOCHS,validation_data=val_datasets,callbacks=callbacks, max_queue_size=10,workers=8,use_multiprocessing=True)
