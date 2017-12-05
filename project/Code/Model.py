# -*- coding: utf-8 -*-
"""
Model Architecture: Refer to the architecture in the report 

Creator: Ajinkya Khamkar

model(): initializes model

fit_(): compiles and runs model
It uses data generator to dynamically generate input set making
it memory effiecient and also randomly samples from the dataset while training
stabilizing training

"""

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Reshape,Dropout,Input, Convolution2D,Dense, Flatten, BatchNormalization,Concatenate, LSTM,merge


class Model_():
    
    def model():
        
        inp = Input(shape=[120,120,3])
        
        C = (Convolution2D(32,(3,3),strides=(2, 2),activation = 'relu',padding = 'same',kernel_initializer='glorot_normal'))(inp)
        C = (Dropout(0.5))(C)
        C = (BatchNormalization())(C)
        
        C = (Convolution2D(64,(3,3),strides=(2, 2),activation = 'relu',padding = 'same',kernel_initializer='glorot_normal'))(C)
        C = (Dropout(0.5))(C)
        C = (BatchNormalization())(C)
        C = (Convolution2D(128,(3,3),strides=(2, 2),activation = 'relu',padding = 'same',kernel_initializer='glorot_normal'))(C)
        C = (Dropout(0.5))(C)
        C = (BatchNormalization())(C)
        
        C = (Convolution2D(256,(3,3),strides=(2, 2),activation = 'relu',padding = 'same',kernel_initializer='glorot_normal'))(C)
        C = (Dropout(0.5))(C)
        C = (BatchNormalization())(C)
        G1 = (Flatten())(C)
        D1 = (Dense(128,activation = 'relu'))(G1)
        D1 = (Dropout(0.4))(D1)
        D1 = (Dense(64,activation = 'relu'))(D1)
        D1 = (Dropout(0.4))(D1)
        X1 = (Dense(1,name='X1'))(D1)
        X2 = (Dense(1,name='X2'))(D1)
        X3 = (Dense(1,name='X3'))(D1)
        X4 = (Dense(1,name='X4'))(D1)
        D2 = (Dense(16,activation = 'softmax',name='D2'))(D1)
        
        A = Concatenate()([G1,X1,X2,X3,X4,D2])
        A = Reshape([1,16404])(A)
        
        L1 = LSTM(16,return_sequences=True)(A)
        L2 = LSTM(4,return_sequences=True,name = 'L2')(L1)
    
        
        model = Model(inputs = inp, outputs = [X1,X2,X3,X4,L2,D2])
        
        print (model.summary())
        
        return (model)
    
    
    def fit_(x_train,y_train,rolledbbox,model):
    
        def train_generator():
            while 1:
                i = np.random.randint(0,x_train.shape[0]-128)
                x = x_train[i:i + 127,:,:,:]
                y = y_train [i:i + 127,:]
                r = rolledbbox[i:i + 127,:]
                yield x,[y[:,0],y[:,1],y[:,2],y[:,3],r,y[:,4:]]
        
        def validation_generator():
            while 1:
                i = np.random.randint(0,x_train.shape[0]-128)
                x = x_train[i:i + 127,:,:,:]
                y = y_train [i:i + 127,:]
                r = rolledbbox[i:i + 127,:]
                yield x,[y[:,0],y[:,1],y[:,2],y[:,3],r,y[:,4:]]
        
        model.compile(optimizer='adam',loss=['mean_squared_error','mean_squared_error','mean_squared_error','mean_squared_error','mean_squared_error','categorical_crossentropy'],metrics={'X1': 'mse','X2': 'mse','X3': 'mse','X4': 'mse','L2': 'mse','D2': 'accuracy'})
        
        history = model.fit_generator(train_generator(),
                            steps_per_epoch=100,
                            epochs = 10,
                            validation_data = validation_generator(),
                            validation_steps=20)
        
        #save model loss and acuracy history
        
        loss_history = history.history["val_loss"]
        numpy_loss_history = np.array(loss_history)
        np.savetxt("val_loss_history.txt", numpy_loss_history, delimiter=",")

        loss_history = history.history["loss"]
        numpy_loss_history = np.array(loss_history)
        np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")

        #save model
        
        model.save('model1.h5')
        
        return (model)
        