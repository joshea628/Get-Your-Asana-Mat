import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import matplotlib as mpl
from skimage import io, color, filters
from skimage.filters import sobel
from skimage.feature._canny import canny

plt.style.use('seaborn')
mpl.rcParams['figure.dpi'] = 100

def display_image(image, ax, title, square_size=43):
    '''
    Reshapes image vectors into array and displays the image in greyscale
    '''
    display = image.reshape(square_size,square_size)
    ax.imshow(display,cmap='gray')
    ax.set_title(title, fontsize=16)
    ax.set_axis_off()
    return ax

def avg_pixel_intensity(data, classname):
    '''
    Creates and displays the average pixel intensity image for a class
    '''
    avg_image = data.mean(axis=0)
    title = f'Average {classname}'
    fig, ax = plt.subplots(1)
    display_image(avg_image, ax, title)
    plt.savefig(f'../images/avg_pixel_intensity_{classname}.png')
    return ax

def histograms_of_pixel_intensities(data, classname):
    '''
    Creates a histogram of pixel intensities for the average image of a class
    '''
    avg_image = data.mean(axis=0)
    fig, ax = plt.subplots(1,figsize=(6,6))
    ax.hist(avg_image)
    ax.set_xlabel('Pixel Intensities', fontsize=14)
    ax.set_ylabel('Frequency in Image', fontsize=14)
    ax.set_title(f'Average {classname} Histogram', fontsize=16)
    plt.savefig(f'../images/avg_histogram_{classname}.png')
    return ax

def apply_filter(data, classname, img_filter=sobel,square_size=43):
    avg_image = data.mean(axis=0)
    shaped_avg_image = avg_image.reshape(square_size,square_size)
    filter_image = img_filter(shaped_avg_image, sigma=4) #un comment for canny filter
    fig, ax = plt.subplots(1)
    title = f'{classname} With Canny Filter'
    display_image(filter_image, ax, title)
    plt.savefig(f'../images/avg_canny_{classname}.png')
    return ax

if __name__ == '__main__':
    poses = ['downdog', 'mountain']
    for pose in poses:
        filepath = f'{pose}.npy'
        data = np.load(filepath)
        #avg_pixel_intensity(data, pose)
        #histograms_of_pixel_intensities(data, pose)
        apply_filter(data, pose, img_filter=canny)
        plt.show()

