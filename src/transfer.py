from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt

class TransferModel(object):
    def __init__(self, batch_size, nb_classes, input_shape, 
                    training_images, testing_images)
        self.batch_size = batch_sizes
        self.classes = nb_classes
        self.input_shape = input_shape
        self.training_images = training_images
        self.testing_images = testing_images
        self.steps_per_epoch = int(training_images/batch_size) + 1
        self.validation_steps = int(testing_images/batch_size)+ 1

    def create_transfer_model(self, weights='imagenet', model=Xception, dropout=0.5):
        """
        Creates model without top and attaches new head to it
        Args:
            input_size (tuple(int, int, int)): 3-dimensional size of input to model
            n_categories (int): number of classification categories
            weights (str or arg): weights to use for model
            model (keras Sequential model): model to use for transfer
        Returns:
            keras Sequential model: model with new head
            """
        if model == Xception:
            base_model = model(weights=weights,
                                include_top=False,
                                input_shape=self.input_size)
        elif model == DenseNet169:
            base_model = model(weights=weights,
                                include_top=True,
                                input_tensor=None,
                                input_shape=input_size,
                                pooling=None,
                                classes=1000)      
        self.model = base_model.output
        self.model = GlobalAveragePooling2D()(self.mod)
        self.model = Dropout(dropout)(self.mod)
        predictions = Dense(self.classes, activation='softmax')(self.model)
        self.model = Model(inputs=base_model.input, outputs=predictions)
        return self.model

    def print_model_properties(self, indices = 0):
        '''
        Prints the properties of individual layers that make up the model
        and whether they are trainable or not.
        '''
        for i, layer in enumerate(self.model.layers[indices:]):
            print("Layer {} | Name: {} | Trainable: {}".format(i+indices, 
                    layer.name, layer.trainable))
        return None
        
    def change_trainable_layers(self, trainable_index):
        '''
        Updates layers to trainable for given index and forward.
        '''
        for layer in self.model.layers[:trainable_index]:
            layer.trainable = False
        for layer in self.model.layers[trainable_index:]:
            layer.trainable = True
        return None
    
    def compile(self, optimizer=Adam(), learning_rate = 0.001, 
                loss=['categorical_crossentropy'], metrics=['accuracy'])
        self.model.compile(optimizer=optimizer, loss, metrics)
        return self
    
    def fit(self, X_train, y_train, X_test, y_test, epochs):
        mdl_check_trans = ModelCheckpoint(filepath='../models/best_trans_model.hdf5', 
                                            monitor='val_accuracy', 
                                            mode='max',save_best_only=True)
        self.model.fit(x=X_train,y=y_train, 
                            epochs=epochs, 
                            steps_per_epoch=self.steps_per_epoch, 
                            validation_data=(X_test, y_test),
                            validation_steps=self.validation_steps,
                            callbacks=[mdl_check_trans])
        return self

    def evaluate(self, X_test, y_test):
        metrics = self.model.evaluate(x=X_test, y=y_test)
        return metrics

def encode_y(y):
    '''
    Encodes y labels as numbers to be sent through model
    '''
    le = LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)
    return y_encoded

if __name__=='__main__':
    batch_size = 32  
    nb_classes = 16
    epochs = 25
    input_shape = (100, 100, 3)
    training_images = 2009
    testing_images = 894
    #load data
    X = np.load('../data/X_trans.npy')
    y = np.load('../data/y_trans.npy')
    #encode data to run through CNN
    y_encoded = encode_y(y)
    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded,random_state=0)
    # #create holdout
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    y_train = to_categorical(y_train, num_classes=nb_classes)
    y_test = to_categorical(y_test, nb_classes) 
    y_val = to_categorical(y_val, nb_classes) 
    trans = TransferModel(batch_size=batch_size, nb_classes=nb_classes, input_shape=input_shape, 
                            training_images=training_images, testing_images=testing_images)
    trans.change_trainable_layers(132)
    trans.compile(optimizer=Adam(), learning_rate = 0.001, 
                    loss=['categorical_crossentropy'], metrics=['accuracy']))
    trans.fit(X_train,y_train, X_test, y_test, epochs=epochs)
    metrics = trans.evaluate(X_test, y_test)