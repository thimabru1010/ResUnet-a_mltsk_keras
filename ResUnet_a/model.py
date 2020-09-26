import tensorflow.keras.models as KM
import tensorflow.keras as KE
import tensorflow.keras.layers as KL


class Resunet_a(object):
    def __init__(self, input_shape, num_classes, args, inputs=None):
        self.num_classes = num_classes
        self.img_height, self.img_width, self.img_channel = input_shape
        self.args = args
        self.inputs = inputs
        self.model = self.build_model_ResUneta()

    def build_model_ResUneta(self):
        def ResBlock(input, filter, kernel_size, dilation_rates, stride):
            def branch(dilation_rate):
                x = KL.BatchNormalization()(input)
                x = KL.Activation('relu')(x)
                x = KL.Conv2D(filter, kernel_size, strides=stride,
                              dilation_rate=dilation_rate, padding='same')(x)
                x = KL.BatchNormalization()(x)
                x = KL.Activation('relu')(x)
                x = KL.Conv2D(filter, kernel_size, strides=stride,
                              dilation_rate=dilation_rate, padding='same')(x)
                return x
            out = []
            for d in dilation_rates:
                out.append(branch(d))
            if len(dilation_rates) > 1:
                out = KL.Add()(out)
            else:
                out = out[0]
            return out

        def PSPPooling(input, filter):
            x1 = KL.MaxPooling2D(pool_size=(1, 1))(input)
            x2 = KL.MaxPooling2D(pool_size=(2, 2))(input)
            x3 = KL.MaxPooling2D(pool_size=(4, 4))(input)
            if self.img_width >= 256:
                x4 = KL.MaxPooling2D(pool_size=(8, 8))(input)
            x1 = KL.Conv2D(int(filter/4), (1, 1))(x1)
            x2 = KL.Conv2D(int(filter/4), (1, 1))(x2)
            x3 = KL.Conv2D(int(filter/4), (1, 1))(x3)
            if self.img_width >= 256:
                x4 = KL.Conv2D(int(filter/4), (1, 1))(x4)
            x1 = KL.UpSampling2D(size=(1, 1))(x1)
            x2 = KL.UpSampling2D(size=(2, 2))(x2)
            x3 = KL.UpSampling2D(size=(4, 4))(x3)
            if self.img_width >= 256:
                x4 = KL.UpSampling2D(size=(8, 8))(x4)
                x = KL.Concatenate()([x1, x2, x3, x4, input])
            else:
                x = KL.Concatenate()([x1, x2, x3, input])
            x = KL.Conv2D(filter, (1, 1))(x)
            return x

        def combine(input1, input2, filter):
            x = KL.Activation('relu')(input1)
            x = KL.Concatenate()([x, input2])
            x = KL.Conv2D(filter, (1, 1))(x)
            return x

        if self.inputs is None:
            self.inputs = KE.Input(shape=(self.img_height,
                          self.img_width, self.img_channel))

        # Encoder
        c1 = x = KL.Conv2D(32, (1, 1), strides=(1, 1), dilation_rate=1)(self.inputs)
        c2 = x = ResBlock(x, 32, (3, 3), [1, 3, 15, 31], (1, 1))
        x = KL.Conv2D(64, (1, 1), strides=(2, 2))(x)
        c3 = x = ResBlock(x, 64, (3, 3), [1, 3, 15, 31], (1, 1))
        x = KL.Conv2D(128, (1, 1), strides=(2, 2))(x)
        c4 = x = ResBlock(x, 128, (3, 3), [1, 3, 15], (1, 1))
        x = KL.Conv2D(256, (1, 1), strides=(2, 2))(x)
        c5 = x = ResBlock(x, 256, (3, 3), [1, 3, 15], (1, 1))
        x = KL.Conv2D(512, (1, 1), strides=(2, 2))(x)
        c6 = x = ResBlock(x, 512, (3, 3), [1], (1, 1))
        x = KL.Conv2D(1024, (1, 1), strides=(2, 2))(x)
        x = ResBlock(x, 1024, (3, 3), [1], (1, 1))

        x = PSPPooling(x, 1024)

        # Decoder
        x = KL.Conv2D(512, (1, 1))(x)
        x = KL.UpSampling2D()(x)
        x = combine(x, c6, 512)
        x = ResBlock(x, 512, (3, 3), [1], 1)
        x = KL.Conv2D(256, (1, 1))(x)
        x = KL.UpSampling2D()(x)
        x = combine(x, c5, 256)
        x = ResBlock(x, 256, (3, 3), [1, 3, 15], 1)
        x = KL.Conv2D(128, (1, 1))(x)
        x = KL.UpSampling2D()(x)
        x = combine(x, c4, 128)
        x = ResBlock(x, 128, (3, 3), [1, 3, 15], 1)
        x = KL.Conv2D(64, (1, 1))(x)
        x = KL.UpSampling2D()(x)
        x = combine(x, c3, 64)
        x = ResBlock(x, 64, (3, 3), [1, 3, 15, 31], 1)
        x = KL.Conv2D(32, (1, 1))(x)
        x = KL.UpSampling2D()(x)
        x = combine(x, c2, 32)
        x = ResBlock(x, 32, (3, 3), [1, 3, 15, 31], 1)

        x_comb = combine(x, c1, 32)
        x_psp = PSPPooling(x_comb, 32)

        if not self.args.multitasking:
            x = KL.Conv2D(self.num_classes, (1, 1))(x_psp)
            x = KL.Activation('softmax')(x)
            model = KM.Model(inputs=inputs, outputs=x)
        else:
            # Models' output
            out = []

            # Segmentation
            x_seg = KL.Conv2D(32, (3, 3), activation='relu', padding='same',
                              name='seg1')(x_psp)
            x_seg = KL.Conv2D(32, (3, 3), activation='relu', padding='same',
                              name='seg2')(x_seg)
            x_seg = KL.Conv2D(self.num_classes, (1, 1), padding='valid',
                              name='seg3')(x_seg)
            out_seg = KL.Activation('softmax', name='seg')(x_seg)
            out.append(out_seg)

            # Boundary
            if self.args.bound:
                x_bound = KL.Conv2D(32, (3, 3), activation='relu',
                                    padding='same')(x_psp)
                x_bound = KL.Conv2D(self.num_classes, (1, 1),
                                    padding='valid')(x_bound)
                out_bound = KL.Activation('sigmoid', name='bound')(x_bound)
                out.append(out_bound)

            # Distance
            if self.args.dist:
                x_dist = KL.Conv2D(32, (3, 3), activation='relu',
                                   padding='same')(x_comb)
                x_dist = KL.Conv2D(32, (3, 3), activation='relu',
                                   padding='same')(x_dist)
                x_dist = KL.Conv2D(self.num_classes, (1, 1),
                                   padding='valid')(x_dist)
                out_dist = KL.Activation('softmax', name='dist')(x_dist)
                out.append(out_dist)

            # Color
            if self.args.color:
                out_color = KL.Conv2D(3, (1, 1), activation='sigmoid',
                                      padding='valid', name='color')(x_comb)
                out.append(out_color)

            # out = [out_seg, out_bound, out_dist, out_color]
            if self.args.gpu_parallel:
                return self.inputs, out
                # model=KM.Model(inputs=inputs,outputs=out)
            else:
                model = KM.Model(inputs=self.inputs, outputs={'seg': out_seg, 'bound': out_bound, 'dist': out_bound,
                                                              'color': out_color})

        return model
