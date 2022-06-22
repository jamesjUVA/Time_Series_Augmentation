from tensorflow import keras
import numpy as np


def augment(x,config, aug_groups,verbose=False):
    choice = [0,1]
    probs = [0,1]
    choice_probs = 'choice_probs'
    if choice_probs in config:
        choice = config[choice_probs][0]
        probs = config[choice_probs][1]
    
    aug_groups_key_copy=list(aug_groups.keys()).copy()
    np.random.shuffle(aug_groups_key_copy)
    count=np.random.choice(choice,p=probs)
    count_=0
    passes=0
    last_aug=''
    if verbose:
        print('Num Augs')
        print(count)
    while count_<count and passes<10:
        for aug_ in aug_groups_key_copy:
            if count_==count:
                return x
            else:
                
                if np.random.random() < config['aug_'+aug_+'_prob']:
                    if last_aug == aug_:
                        continue
                    last_aug = aug_  # Don't execute same aug type twice
                    aug_func=np.random.choice(aug_groups[aug_])
                    if verbose:
                        print(aug_)
                        print(aug_func.__name__)

                    if aug_func.__name__!="rotation":
                        x = aug_func(x,config['aug_'+aug_+'_prob'])
                    else:
                        x = aug_func(x)    
                    count_=count_+1
                passes=passes+1
    return x




def build_model(input_shape, nb_classes, model_type = 'resnet', n_feature_maps=32, dropout=0, dense_layer = 0, depth = 6, num_kernel_sizes = 3, kernel_size = 41, bottleneck=32 ):
    if model_type == 'resnet':
        
        input_layer = keras.layers.Input(input_shape)

        # BLOCK 1
        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)



    if model_type == 'inception':

        use_residual=True
        use_bottleneck=True 
        depth=depth 
        kernel_size=kernel_size #51 #41
        nb_filters = n_feature_maps
        kernel_size = kernel_size - 1
        bottleneck_size = bottleneck #32

        def _inception_module(input_tensor, stride=1, activation='linear'):

            
            
            if use_bottleneck and int(input_tensor.shape[-1]) > 1:
                input_inception = keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                                      padding='same', activation=activation, use_bias=False)(input_tensor)               
            else:
                input_inception = input_tensor

            # kernel_size_s = [3, 5, 8, 11, 17]
            kernel_size_s = [int(kernel_size // (2 ** i)) for i in range(num_kernel_sizes)]
            #kernel_size_s = [kernel_size // (2 ** i) for i in range(4)]
            conv_list = []

            for i in range(len(kernel_size_s)):
                conv_list.append(keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                                     strides=stride, padding='same', activation=activation, use_bias=False)(
                    input_inception))

            max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

            conv_6 = keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                         padding='same', activation=activation, use_bias=False)(max_pool_1)

            conv_list.append(conv_6)

            x = keras.layers.Concatenate(axis=2)(conv_list)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation(activation='relu')(x)
            return x

        def _shortcut_layer(input_tensor, out_tensor):
            shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                             padding='same', use_bias=False)(input_tensor)
            shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

            x = keras.layers.Add()([shortcut_y, out_tensor])
            x = keras.layers.Activation('relu')(x)
            return x

        
        print(input_shape)
        input_layer = keras.layers.Input(input_shape)
        print(input_layer)


        x = input_layer
        input_res = input_layer
        print(x)
        for d in range(depth):

            x = _inception_module(x)

            if use_residual and d % 3 == 2:
                x = _shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

 
        if (dense_layer > 0) and (dropout>.1):
            dropout_layer = keras.layers.Dropout(dropout)(gap_layer)
            dense_layer = keras.layers.Dense(dense_layer, activation='relu')(dropout_layer)
            dropout_layer2 = keras.layers.Dropout(dropout)(dense_layer)
            output_layer = keras.layers.Dense(nb_classes, activation='softmax')(dropout_layer2)
        elif (dropout>0.1) and (dense_layer==0):
            dropout_layer = keras.layers.Dropout(dropout)(gap_layer)
            output_layer = keras.layers.Dense(nb_classes, activation='softmax')(dropout_layer)
        elif (dense_layer > 0) and (dropout<.1):
            dense_layer = keras.layers.Dense(dense_layer, activation='relu')(gap_layer)
            output_layer = keras.layers.Dense(nb_classes, activation='softmax')(dense_layer)
        else:
            output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)


        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
              

    if model_type == 'inceptionFCN':

        use_residual=True
        use_bottleneck=True 
        depth=depth 
        kernel_size=kernel_size - 1 #51 #41
        nb_filters = n_feature_maps
        kernel_size = kernel_size - 1
        bottleneck_size = bottleneck #32

        def _inception_module(input_tensor, stride=1, activation='linear'):

            if use_bottleneck and int(input_tensor.shape[-1]) > 1:
                input_inception = keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                                      padding='same', activation='linear', use_bias=False)(input_tensor)
            else:
                input_inception = input_tensor

            # kernel_size_s = [10, 20]
            #if kernel_size <40: 
            kernel_size_s = [int(kernel_size // (2 ** i)) for i in range(2,0,-1)]
            #else:
            #    kernel_size_s = [int(kernel_size // (2 ** i)) for i in range(3,0,-1)]
            conv_list = []

            for i in range(len(kernel_size_s)):
                conv_list.append(keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                                     strides=stride, padding='same', activation=activation, use_bias=False)(input_inception))

            max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

            conv_6 = keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                         padding='same', activation=activation, use_bias=False)(max_pool_1)

            conv_list.append(conv_6)

            x = keras.layers.Concatenate(axis=2)(conv_list)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation(activation='relu')(x)
            return x

        def _shortcut_layer(input_tensor, out_tensor):
            
            kernel_sizes = [1,3,3,5,5,1]         
            factor = out_tensor.shape[-1]
            print(factor)
            for i in range(len(kernel_sizes)):
                shortcut_y = keras.layers.Conv1D(filters=int(factor), kernel_size=kernel_sizes[i],
                                             padding='same', use_bias=False)(input_tensor)
                shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
                x = keras.layers.Add()([shortcut_y, out_tensor])
                x = keras.layers.Activation('relu')(x)

            return x


        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(depth):

            x = _inception_module(x)

            if use_residual and d % 3 == 2:
                x = _shortcut_layer(input_res, x)
                input_res = x
        #max_pool_layer = keras.layers.MaxPooling1D(3)(x)
        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        #dropout_layer = keras.layers.Dropout(rate=0.5)(gap_layer)
        if (dense_layer > 0) and (dropout>.1):
            dropout_layer = keras.layers.Dropout(dropout)(gap_layer)
            dense_layer = keras.layers.Dense(dense_layer, activation='relu')(dropout_layer)
            dropout_layer2 = keras.layers.Dropout(dropout)(dense_layer)
            output_layer = keras.layers.Dense(nb_classes, activation='softmax')(dropout_layer2)
        elif (dropout>0.1) and (dense_layer==0):
            dropout_layer = keras.layers.Dropout(dropout)(gap_layer)
            output_layer = keras.layers.Dense(nb_classes, activation='softmax')(dropout_layer)
        elif (dense_layer > 0) and (dropout<.1):
            dense_layer = keras.layers.Dense(dense_layer, activation='relu')(gap_layer)
            output_layer = keras.layers.Dense(nb_classes, activation='softmax')(dense_layer)
        else:
            output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)      
        
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)


    if model_type == 'inceptionFCNMulti':

        use_residual=True
        use_bottleneck=True 
        depth=depth 
        kernel_size=kernel_size - 1 #51 #41
        nb_filters = n_feature_maps
        kernel_size = kernel_size - 1
        bottleneck_size = bottleneck #32

        def _inception_module(input_tensor, stride=1, activation='linear'):

            if use_bottleneck and int(input_tensor.shape[-1]) > 1:
                input_inception = keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                                      padding='same', activation='linear', use_bias=False)(input_tensor)
            else:
                input_inception = input_tensor

            # kernel_size_s = [10, 20]
            #if kernel_size <40: 
            kernel_size_s = [int(kernel_size // (2 ** i)) for i in range(2,0,-1)]
            #else:
            #    kernel_size_s = [int(kernel_size // (2 ** i)) for i in range(3,0,-1)]
            conv_list = []

            for i in range(len(kernel_size_s)):
                conv_list.append(keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                                     strides=stride, padding='same', activation=activation, use_bias=False)(input_inception))

            max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

            conv_6 = keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                         padding='same', activation=activation, use_bias=False)(max_pool_1)

            conv_list.append(conv_6)

            x = keras.layers.Concatenate(axis=2)(conv_list)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation(activation='relu')(x)
            return x

        def _shortcut_layer(input_tensor, out_tensor):
            
            kernel_sizes = [1,3,3,5,5,1]  #1,3,3,5,5,1 original         
            factor = out_tensor.shape[-1]
            print(factor)
            for i in range(len(kernel_sizes)):
                shortcut_y = keras.layers.Conv1D(filters=int(factor), kernel_size=kernel_sizes[i],
                                             padding='same', use_bias=False)(input_tensor)
                shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
                x = keras.layers.Add()([shortcut_y, out_tensor])
                x = keras.layers.Activation('relu')(x)

            return x


        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer
        channel_list = []

     
    

        for d in range(depth):
            if d==0:
                for i in range(x.shape[2]):
                    channel_list.append(_inception_module(keras.backend.expand_dims(x[:,:,i])))
                x = keras.layers.Concatenate(axis=2)(channel_list)
            else:
                x = _inception_module(x)

                if use_residual and d % 3 == 2:
                    x = _shortcut_layer(input_res, x)
                    input_res = x
        #max_pool_layer = keras.layers.MaxPooling1D(3)(x)
        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        #dropout_layer = keras.layers.Dropout(rate=0.5)(gap_layer)
        if (dense_layer > 0) and (dropout>.1):
            dropout_layer = keras.layers.Dropout(dropout)(gap_layer)
            dense_layer = keras.layers.Dense(dense_layer, activation='relu')(dropout_layer)
            dropout_layer2 = keras.layers.Dropout(dropout)(dense_layer)
            output_layer = keras.layers.Dense(nb_classes, activation='softmax')(dropout_layer2)
        elif (dropout>0.1) and (dense_layer==0):
            dropout_layer = keras.layers.Dropout(dropout)(gap_layer)
            output_layer = keras.layers.Dense(nb_classes, activation='softmax')(dropout_layer)
        elif (dense_layer > 0) and (dropout<.1):
            dense_layer = keras.layers.Dense(dense_layer, activation='relu')(gap_layer)
            output_layer = keras.layers.Dense(nb_classes, activation='softmax')(dense_layer)
        else:
            output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)      
        
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        
        
        
    if model_type == 'inceptionFCNMulti2':

        use_residual=True
        use_bottleneck=True 
        depth=depth 
        kernel_size=kernel_size - 1 #51 #41
        nb_filters = n_feature_maps
        kernel_size = kernel_size - 1
        bottleneck_size = bottleneck #32

        def _inception_module(input_tensor, stride=1, activation='linear'):

            if use_bottleneck and int(input_tensor.shape[-1]) > 1:
                input_inception = keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                                      padding='same', activation='linear', use_bias=False)(input_tensor)
            else:
                input_inception = input_tensor

            # kernel_size_s = [10, 20]
            #if kernel_size <40: 
            kernel_size_s = [int(kernel_size // (2 ** i)) for i in range(2,0,-1)]
            #else:
            #    kernel_size_s = [int(kernel_size // (2 ** i)) for i in range(3,0,-1)]
            conv_list = []

            for i in range(len(kernel_size_s)):
                conv_list.append(keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                                     strides=stride, padding='same', activation=activation, use_bias=False)(input_inception))

            max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

            conv_6 = keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                         padding='same', activation=activation, use_bias=False)(max_pool_1)

            conv_list.append(conv_6)

            x = keras.layers.Concatenate(axis=2)(conv_list)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation(activation='relu')(x)
            return x

        
        def _inception_module_lower(input_tensor, stride=1, activation='linear'):

            if use_bottleneck and int(input_tensor.shape[-1]) > 1:
                input_inception = keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                                      padding='same', activation='linear', use_bias=False)(input_tensor)
            else:
                input_inception = input_tensor

            # kernel_size_s = [10, 20]
            #if kernel_size <40: 
            kernel_size_s = [int(kernel_size // (2 ** i)) for i in range(2,0,-1)]
            #else:
            #    kernel_size_s = [int(kernel_size // (2 ** i)) for i in range(3,0,-1)]
            conv_list = []

            for i in range(len(kernel_size_s)):
                conv_list.append(keras.layers.Conv1D(filters=int(nb_filters*.25), kernel_size=kernel_size_s[i],
                                                     strides=stride, padding='same', activation=activation, use_bias=False)(input_inception))

            max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

            conv_6 = keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                         padding='same', activation=activation, use_bias=False)(max_pool_1)

            conv_list.append(conv_6)

            x = keras.layers.Concatenate(axis=2)(conv_list)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation(activation='relu')(x)
            return x
        
        
        def _shortcut_layer(input_tensor, out_tensor):
            
            kernel_sizes = [1,3,3,5,5,1]  #1,3,3,5,5,1 original         
            factor = out_tensor.shape[-1]
            print(factor)
            for i in range(len(kernel_sizes)):
                shortcut_y = keras.layers.Conv1D(filters=int(factor), kernel_size=kernel_sizes[i],
                                             padding='same', use_bias=False)(input_tensor)
                shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
                x = keras.layers.Add()([shortcut_y, out_tensor])
                x = keras.layers.Activation('relu')(x)

            return x


        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer
        channel_list = []

     
        

        for d in range(depth):
            
            if d == 0:
                for i in range(x.shape[2]):
                    xi=_inception_module_lower(keras.backend.expand_dims(x[:,:,i]),stride=1)
                    xi_res = xi
                    xi=_inception_module_lower(xi)
                    xi=_inception_module_lower(xi)
                    xi = _shortcut_layer(xi_res, xi)
                    channel_list.append(xi)
                x = keras.layers.Concatenate(axis=2)(channel_list)
            else:
                x = _inception_module(x)

                if use_residual and (d-1) % 3 == 2:
                    x = _shortcut_layer(input_res, x)
                    input_res = x
        #max_pool_layer = keras.layers.MaxPooling1D(3)(x)
        
        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        #dropout_layer = keras.layers.Dropout(rate=0.5)(gap_layer)
        if (dense_layer > 0) and (dropout>.1):
            dropout_layer = keras.layers.Dropout(dropout)(gap_layer)
            dense_layer = keras.layers.Dense(dense_layer, activation='relu')(dropout_layer)
            dropout_layer2 = keras.layers.Dropout(dropout)(dense_layer)
            output_layer = keras.layers.Dense(nb_classes, activation='softmax')(dropout_layer2)
        elif (dropout>0.1) and (dense_layer==0):
            dropout_layer = keras.layers.Dropout(dropout)(gap_layer)
            output_layer = keras.layers.Dense(nb_classes, activation='softmax')(dropout_layer)
        elif (dense_layer > 0) and (dropout<.1):
            dense_layer = keras.layers.Dense(dense_layer, activation='relu')(gap_layer)
            output_layer = keras.layers.Dense(nb_classes, activation='softmax')(dense_layer)
        else:
            output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)      
        
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        
    return model
## Generator
        
class DataGeneratorClassify(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x_train, y_train, x_test, y_test, aug_groups, batch_size=64, augment=True, shuffle=True, config={}):
        # Augs [[minWarp,maxWarp],[minCrop,maxCrop],[minScale,maxScale],[jitter]]
        # Warp is real proportion of trace length 0.1:20
        # crop is up to 50% shift of data cropped off of trace for [-10,10]
        # scale is up to 35% increase at 10 so [-10,10]
        # jitter is max of .005 * random normal * [0,10] 
        'Initialization'
        self.x_train= x_train
        self.y_train= y_train
        self.aug_groups = aug_groups
        nb_classes = len(np.unique(y_train))
        self.nb_classes = nb_classes
        
        # transform the labels from integers to one hot vectors

        ##y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
        ## save orignal y because later we will use binary
        ##y_true = np.argmax(y_test, axis=1)

        if len(x_train.shape) == 2:  # if univariate
            # add a dimension to make it multivariate with one dimension 
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        self.input_shape = x_train.shape[1:]
              

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment=augment
        self.config=config
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.x_train.shape[0]) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.x_train.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indicies):
        #Get y
        y = self.y_train[indicies]
        x_mat_temp = self.x_train[indicies]
        x = np.zeros((x_mat_temp.shape[0],self.input_shape[0],self.input_shape[1]))
        if self.augment==False:
            return x_mat_temp,y
        else:
            
            for i in range(0,x_mat_temp.shape[0]):
            #print('patid:'+str(pat_id['PID']))
                x[i,:]=np.expand_dims(augment(np.squeeze(x_mat_temp[i]), self.config, self.aug_groups,verbose=False),axis=1)
            return x,y

        
class DataGeneratorClassifyMultivariate(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x_train, y_train, x_test, y_test, aug_groups, batch_size=64, augment=True, shuffle=True, config={}):
        # Augs [[minWarp,maxWarp],[minCrop,maxCrop],[minScale,maxScale],[jitter]]
        # Warp is real proportion of trace length 0.1:20
        # crop is up to 50% shift of data cropped off of trace for [-10,10]
        # scale is up to 35% increase at 10 so [-10,10]
        # jitter is max of .005 * random normal * [0,10] 
        'Initialization'
        self.x_train= x_train
        self.y_train= y_train
        self.aug_groups = aug_groups
        nb_classes = len(np.unique(y_train))
        self.nb_classes = nb_classes
        
        # transform the labels from integers to one hot vectors

        ##y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
        ## save orignal y because later we will use binary
        ##y_true = np.argmax(y_test, axis=1)

        if len(x_train.shape) == 2:  # if univariate
            # add a dimension to make it multivariate with one dimension 
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        self.input_shape = x_train.shape[1:]
              

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment=augment
        self.config=config
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.x_train.shape[0]) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.x_train.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indicies):
        #Get y
        y = self.y_train[indicies]
        x_mat_temp = self.x_train[indicies]
        x = np.zeros((x_mat_temp.shape[0],self.input_shape[0],self.input_shape[1]))
        if self.augment==False:
            return x_mat_temp,y
        else:
            
            for i in range(0,x_mat_temp.shape[0]):
            #print('patid:'+str(pat_id['PID']))
                x[i,:]=augment(np.squeeze(x_mat_temp[i]), self.config, self.aug_groups,verbose=False)
            return x,y