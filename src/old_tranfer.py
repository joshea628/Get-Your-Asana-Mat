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
from tensorflow.keras.preprocessing import ImageDataGenerator

def transfer_model(input_size, n_categories, weights = 'imagenet', model=Xception):
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
    input_shape = (299, 299, 3)
    train_dg = ImageDataGenerator(preprocessing_function=preprocessing,
                                                brightness_range=[0.2, 0.8],
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
    test_dg = ImageDataGenerator(preprocessing_function=preprocessing)
    test_generator = test_dg.flow_from_directory('../data/testing',
                                                    target_size=(299,299),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)
    
    transfer_model = transfer_model(input_shape, nb_classes)
    change_trainable_layers(transfer_model, 132)
    #print_model_properties(transfer_model)
    transfer_model.compile(optimizer=Adam(.001), loss=['categorical_crossentropy'], metrics=['accuracy'])
    mdl_check_trans = ModelCheckpoint(filepath='best_trans_model.hdf5', monitor='val_accuracy', 
                                        mode='max',save_best_only=True)
    transfer_model.fit(x=train_generator, 
                        epochs=epochs, 
                        steps_per_epoch=None, 
                        validation_data=(test_generator),
                        validation_steps=None,
                        callbacks=[mdl_check_trans])
    metrics = transfer_model.evaluate(x=test_generator)
    transfer_model.save('bestmodel.h5')
    print(metrics)