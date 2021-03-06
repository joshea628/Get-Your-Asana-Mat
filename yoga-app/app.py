from __future__ import division, print_function
import os
import pickle
import sys
import numpy as np
import pandas as pd
from PIL import Image, ExifTags
from flask import Flask, redirect, render_template, request, url_for, flash
from gevent.pywsgi import WSGIServer
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename

app = Flask(__name__)

def format(filename):
    im = load_img(filename, target_size = (299,299))
    image = preprocess_input(img_to_array(im))
    return np.expand_dims(image, axis=0)

def get_category(img_path,model):
    im = format(img_path)
    pred = model.predict(im)
    top = pred.argsort()[0][::-1][:1]
    top_name = class_names[top][0]
    return f'{top_name}'

@app.route('/', methods=['GET'])
def index():
    # Main page
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():     
   if request.method == 'POST':
        file = request.files['file']
        if file.filename != '':
            filename = secure_filename(file.filename)
            file.save(filename)
            preds = get_category(filename, model)
            os.remove(filename)
        return preds 

if __name__ == '__main__':
    with open ('models/classes.pkl', 'rb') as f:
        class_names = np.array(pickle.load(f))

    model = load_model('models/88.5dmhc.h5')
    print('Model loaded. Start serving...')

    http_server = WSGIServer(('0.0.0.0',8105), app)
    http_server.serve_forever()