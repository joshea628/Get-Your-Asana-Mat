import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import matplotlib as mpl
from skimage import io, color, filters
from skimage.filters import sobel
from skimage.feature._canny import canny
import pandas as pd 
import seaborn as sns

plt.style.use('seaborn')
mpl.rcParams['figure.dpi'] = 500

def display_image(image, ax, title, square_size=43):
    '''
    Reshapes image vectors into array and displays the image in greyscale

    input: image, axis, title of graph, size to reshape image (should match 
    original resize number in pipeline)
    output: image
    '''
    display = image.reshape(square_size,square_size)
    ax.imshow(display,cmap='gray')
    ax.set_title(title, fontsize=24)
    ax.set_axis_off()
    return ax

def avg_pixel_intensity(data, classname):
    '''
    Creates and displays the average pixel intensity image for a class

    input: image vectors in an array, pose name
    output: image of average pixel intensity of entire class
    '''
    avg_image = data.mean(axis=0)
    name = f'{classname}'
    name = name.capitalize()
    title = f'Average {name}'
    fig, ax = plt.subplots(1)
    display_image(avg_image, ax, title)
    plt.savefig(f'../images/avg_pixel_intensity_{classname}.png')

    return ax

def histograms_of_pixel_intensities(data, classname):
    '''
    Creates a histogram of pixel intensities for the average image of a 
    class

    input: image vectors in an array, pose name
    output: histogram of the average pixel intensity of entire class
    '''
    avg_image = data.mean(axis=0)
    fig, ax = plt.subplots(1,figsize=(6,6))
    ax.hist(avg_image)
    ax.set_xlabel('Pixel Intensities', fontsize=18)
    ax.set_ylabel('Frequency in Image', fontsize=18)
    ax.set_title(f'Average {classname} Histogram', fontsize=24)
    plt.savefig(f'../images/avg_histogram_{classname}.png')
    return ax

def apply_filter(data, classname, img_filter=sobel,square_size=43):
    '''
    Creates an image using edge detection from either the Sobel or Canny 
    filter

    input: image vectors in an array, pose name, filter, resize
    output: filtered image of the average pixel intensity of entire class
    '''
    avg_image = data.mean(axis=0)
    shaped_avg_image = data.reshape(square_size,square_size)
    filter_image = img_filter(shaped_avg_image, sigma=4)
    fig, ax = plt.subplots(1)
    name = f'{classname}'
    name = name.capitalize()
    title = f'{name} With Canny Filter'
    display_image(filter_image, ax, title)
    plt.savefig(f'../images/wrong/canny6.png') #avg_canny_{classname}
    return ax

def flatten_and_save_canny(data, square_size=43):
    '''
    applies the canny filter to each image in the dataset, and vectorizes 
    each picture into one line.

    input: data set of images as vector arrays
    output: canny images as vector arrays
    '''
    canny_images = []
    for image in data:
        shaped_image = image.reshape(square_size, square_size)
        filter_image = canny(shaped_image, sigma=4)
        flattened_filtered = np.ravel(filter_image)
        canny_images.append(flattened_filtered)
    return canny_images

def accuracies():
    '''
    Plots accuracies for different models for presentation
    '''
    ax, fig = plt.subplots(1,figsize=(10,6))
    accs = [66.4,79.6]#25,88.5] #
    models = ['Random Forest', 'Logistic Regression'] #'Basic CNN', 'Xception'
    plt.bar(models,accs)
    for i, v in enumerate(accs):
        ax.text(i-.25, 
                    v/accs[i]+100, 
                    accs[i], 
                    fontsize=18)
    plt.xlabel('Model', fontsize=18)
    plt.ylabel('Percent Accuracy', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Accuracy Comparison of 2 Pose Models with 3 PCA Components', fontsize=24)
    plt.savefig('accuracies.png')

if __name__ == '__main__':
    poses = ['downdog','mountain']
    canny_data = []
    for pose in poses:
        filepath = f'../data/{pose}.npy'
        data = np.load(filepath)
        avg_pixel_intensity(data, pose)
        histograms_of_pixel_intensities(data, pose)
        apply_filter(data, pose, img_filter=canny)
  
    accuracies()
    plt.show()

