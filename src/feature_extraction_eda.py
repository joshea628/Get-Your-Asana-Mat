import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

plt.style.use('bmh')

def display_image(image, ax, title):
    ax.imshow(image)
    ax.set_title(title)
    ax.set_axis_off()
    return ax

def avg_pixel_intensity(data, classname):
    avg_image = data.mean(axis=0)
    title = f'Average {classname}'
    fig, ax = plt.subplots(1)
    display_image(avg_image, ax, title)
    return ax 

