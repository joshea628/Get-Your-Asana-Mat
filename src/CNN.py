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

plt.style.use('seaborn')
mpl.rcParams['figure.dpi'] = 100

def def_model(nb_filters, input_shape, kernel_size, pool_size):
    '''
    Returns base CNN model: 2 Layers using softmax and adam.
    '''
    model = Sequential()  
    # Activation Function since this is a multi-classification problem, use softmax.
    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='valid',input_shape=input_shape))
    model.add(Activation('softmax'))  
    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='valid'))
    model.add(Activation('softmax'))
    model.add(MaxPooling2D(pool_size=pool_size))  # decreases size, helps prevent overfitting
    model.add(Dropout(0.5))  # zeros out some fraction of inputs, helps prevent overfitting
    model.add(Flatten())  # necessary to flatten before going into conventional dense layer
    model.add(Dense(32))  # (only) 32 neurons in this layer, really?
    model.add(Activation('softmax'))
    model.add(Dropout(0.5))  # zeros out some fraction of inputs, helps prevent overfitting
    model.add(Dense(nb_classes))  # 16 final nodes (one for each class)
    model.add(Activation('softmax'))
    # optimizers to one of these: 'adam', 'adadelta', 'sgd'
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

if __name__=='__main__':
    #load data
    X = np.load('../data/X_four.npy')
    y= np.load('../data/y_four.npy')
    
    #encode data as INTs to run through CNN
    le = LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)

    # CNN parameters
    batch_size = 32  # number of training samples used at a time to update the weights
    nb_classes = 4    # number of output possibilities
    nb_epoch = 3       # number of passes through the entire train dataset before weights "final"
    img_rows, img_cols = 100, 100   # the size of the images
    nb_filters = 16   # number of convolutional filters to use
    pool_size = (2, 2)  # pooling decreases image size, reduces computation, adds translational invariance
    kernel_size = (4, 4)  # convolutional kernel size, slides over image to learn features
    input_shape = (img_rows, img_cols, 1)

    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded,random_state=0)
    #create holdout
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    
    #convert data for 2DCon
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 3)
    X_train = X_train.astype('float32') # data was uint8 [0-255]
    X_test = X_test.astype('float32')  # data was uint8 [0-255]
    y_train = to_categorical(y_train, num_classes=nb_classes)
    y_test = to_categorical(y_test, nb_classes) 
    y_val = to_categorical(y_val, nb_classes) 

    
    model = def_model(nb_filters, input_shape, kernel_size, pool_size)

    # during fit process watch train and test error simultaneously
    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1]) 

    
