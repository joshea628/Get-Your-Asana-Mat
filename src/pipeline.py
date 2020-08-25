import numpy as np
import os
import io 
from PIL import Image
import matplotlib.pyplot as plt
from requests import get

class ImageManipulation(object): 
    ''' 
    Pipeline to process images

    input: path to image file (string)
    resolution: pixel size (int)

    output:

    methods: greyscale, resize, convert_to_array

    '''
    def __init__(self, path, resolution, pose):
        self.path = path
        self.resolution = resolution
        self.pose = pose
        self.image_paths = []
        self.image_array = []
        self.image_grey = []
        self.image_resized = []

    def _read_images(self):
        '''
        Reads in file and adds images to list
        '''
        with open(self.path) as f:
            self.image_list = f.readlines()
         
        self.image_list = [x.strip() for x in self.image_list]

        for url in self.image_list:
            try:
                response = get(url)
                image_bytes = io.BytesIO(response.content)
                image = Image.open(image_bytes)
                self.image_array.append(image)
            except OSError:
                print('image failed...') 
        return self  
             
    def greyscale(self):
        '''
        Converts image to greyscale
        '''
        self._read_images()
        for image in self.image_array:
            grey_image = image.convert('L')
            self.image_grey.append(grey_image)
        return self 

    def resize(self):
        '''
        resizes image to resolution input
        '''
        for image in self.image_grey:
            self.image_resized.append(image.resize((self.resolution,self.resolution)))
        return self

    def save_images(self):
        '''
        saves images as a numpy array and saves to data directory
        '''
        image_final = []
        for image in self.image_resized:
            image_matrix = np.array(image)
            image_vector = np.ravel(image_matrix)
            image_final.append(image_vector)
        np.save(self.pose, image_final)
        print(f'Images saved as: {self.pose}.npy')
        return self 

if __name__ == "__main__":
    poses = ['downdog','mountain']

    for pose in poses:
        path = f'../data/{pose}.txt'
        process = ImageManipulation(path, 43, pose)
        process.greyscale()
        process.resize()
        process.save_images()
        #print(final)

    
    
