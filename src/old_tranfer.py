from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt

def create_transfer_model(input_size, n_categories, weights = 'imagenet', model=Xception):
    """
    Creates transfer learning model
    """
    if model==Xception:
      base_model = model(weights=weights,
                          include_top=False,
                          input_shape=input_size)
    elif model==Densenet169:
      base_model = model(weights=weights,
                          include_top=True,
                          input_tensor=None,
                          input_shape=None,
                          pooling=None,
                          classes=16)    
    mod = base_model.output
    mod = GlobalAveragePooling2D()(mod)
    mod = Dropout(0.2)(mod)
    predictions = Dense(n_categories, activation='softmax')(mod)
    mod = Model(inputs=base_model.input, outputs=predictions)
    return mod

def print_model_properties(model, indices = 0):
     for i, layer in enumerate(model.layers[indices:]):
        print("Layer {} | Name: {} | Trainable: {}".format(i+indices, layer.name, layer.trainable))
        
def change_trainable_layers(model, trainable_index):
    for layer in model.layers[:trainable_index]:
        layer.trainable = False
    for layer in model.layers[trainable_index:]:
        layer.trainable = True


if __name__=='__main__':
    batch_size = 32  
    nb_classes = 4
    epochs = 100
    img_rows, img_cols = 299, 299
    input_shape = (img_rows, img_cols, 3)
    training_images = 2009
    testing_images = 894
    steps_per_epoch = int(training_images/batch_size) + 1
    validation_steps = int(testing_images/batch_size)+ 1
    #load data
    X = np.load('X_four299.npy')
    y = np.load('y_four299.npy')
    X = X.reshape(X.shape[0], 299, 299, 3)
    #encode data to run through CNN
    le = LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)
    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded)
    #create holdout
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    y_train = to_categorical(y_train, num_classes=nb_classes)
    y_test = to_categorical(y_test, nb_classes) 
    #y_val = to_categorical(y_val, nb_classes) 

    transfer_model = create_transfer_model(input_shape, nb_classes)
    change_trainable_layers(transfer_model, 132)
    #print_model_properties(transfer_model)
    transfer_model.compile(optimizer=Adam(.001), loss=['categorical_crossentropy'], metrics=['accuracy'])
    mdl_check_trans = ModelCheckpoint(filepath='best_trans_model.hdf5', monitor='val_accuracy', 
                                        mode='max',save_best_only=True)
    transfer_model.fit(x=X_train,y=y_train, 
                        epochs=epochs, 
                        steps_per_epoch=None, 
                        validation_data=(X_test, y_test),
                        validation_steps=None,
                        callbacks=[mdl_check_trans])
    metrics = transfer_model.evaluate(x=X_test, y=y_test)
    transfer_model.save('bestmodel.h5')
    print(metrics)