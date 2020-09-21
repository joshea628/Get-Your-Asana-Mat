import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

plt.style.use('seaborn')
mpl.rcParams['figure.dpi'] = 100

def def_model(nb_filters, input_shape, kernel_size, pool_size):
    '''
    Returns base CNN model: 2 Layers using softmax and adam.
    '''
    model = Sequential()  
    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='valid',input_shape=input_shape))
    model.add(Activation('softmax'))  
    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='valid'))
    model.add(Activation('softmax'))
    model.add(MaxPooling2D(pool_size=pool_size))  
    model.add(Dropout(0.5))  
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('softmax'))
    model.add(Dropout(0.5)) 
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

if __name__=='__main__':
    batch_size = 32
    nb_classes = 4
    nb_epoch = 10 
    nb_filters = 4  
    pool_size = (2, 2) 
    kernel_size = (4, 4)
    input_shape = (299, 299, 3)
    train_dg = ImageDataGenerator(brightness_range=[0.2, 0.8],
                                                zoom_range=0.1,
                                                width_shift_range=0.1,
                                                height_shift_range=0.1,
                                                shear_range=0.1,
                                                horizontal_flip=True,
                                                rotation_range=5)
    train_generator = train_dg.flow_from_directory('../data/training',
                                                    target_size=(299,299),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)
    test_dg = ImageDataGenerator()
    test_generator = test_dg.flow_from_directory('../data/testing',
                                                    target_size=(299,299),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)    
    model = def_model(nb_filters, input_shape, kernel_size, pool_size)
    model.fit(x=train_generator, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(test_generator))
    score = model.evaluate(x=test_generator, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1]) 

    
