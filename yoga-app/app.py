from __future__ import division, print_function

import os
import pickle
import sys
import numpy as np
import pandas as pd
from PIL import Image, ExifTags
from flask import Flask, redirect, render_template, request, url_for
#from gevent.pywsgi import WSGIServer
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename


app = Flask(__name__)

def rotate_save(f, file_path):
    try:
        image=Image.open(f)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif=dict(image._getexif().items())

        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)
        image.save(file_path)
        image.close()

    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        image.save(file_path)
        image.close()

        pass

def process_img(filename):
    """
    Loads image from filename, preprocesses it and expands the dimensions because the model predict function expects a batch of images, not one image
    """
    original = load_img(filename, target_size = (299,299))
    numpy_image = preprocess_input( img_to_array(original))
    image_batch = np.expand_dims(numpy_image, axis =0)
    return image_batch

def get_category(img_path,model):
    """
    Uses an image and a model to return the names and the predictions of the top 3 classes
    """
    im =  process_img(img_path)
    preds =  model.predict(im)
    top_3 = preds.argsort()[0][::-1][:3] # sort in reverse order and return top 3 indices
    top_3_names = class_names[top_3]
    top_3_percent = preds[0][[top_3]]*100
    #top_3_text = '<br>'.join([f'{name}: {percent:.2f}%' for name, percent in zip(top_3_names,top_3_percent)])
    return top_3_names

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']


        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        # f.save(file_path)
        rotate_save(f, file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Delete it so we don't clutter our server up
        os.remove(file_path)

        return preds
    return None

if __name__ == '__main__':
    #MODEL_PATH = 
    with open ('models/classes.pkl', 'rb') as f:
        class_names = np.array(pickle.load(f))

    model = load_model('models/87.5_dog_mount_halfmoon_chair.h5')
    #model._make_predict_function()
    print('Model loaded. Start serving...')

    app.run(host='0.0.0.0',port=8105,debug=True)

    # Serve the app with gevent
    # http_server = WSGIServer(('0.0.0.0',5000), app)
    # http_server.serve_forever()