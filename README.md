# Yoga-Classifier

## Background and Motivation

Yoga has been a huge part of my life, it's something that grounds me and has made me a more patient person. When thinking about project ideas, I wanted to do something that applied to my life in some way. While I'm not usually wondering which pose I am doing, I thought it could be super useful for people learning yoga. 

## Data

The images I'm using for this project have been found on the internet, collected by a user: and shared on Reddit, and the rest of the images I collected from a google image search and from friends and classmates. 

The downdog class has __ images.
The mountain class has __ images.
The halfmoon class has __ images.

I created a pipeline to read in the URLs to the images, convert the images to greyscale, and resize to __ pixels. 

Here is a sample of an original image, greyscale image, and a resized image from each class: 

______ insert reg, grey image, resized image here 

Since yoga is all about shapes, color should not affect the classification at all. Not only does greyscaling and resizing the data make the data smaller, but it makes it easier to work with as well. 

## EDA

Now that the data is grey and easy to work with, it's time for feature extraction! How can I train my model to set it up for success? 

In order to get a feel for the data and for how the models I plan to use would work, I decided to start with only two poses to work with: Downdog and Mountain.

The first thing I did was take a look at the mean pixel intensities for each of the first two classes. 

<p float="middle">
    <img src="images/avg_pixel_intensity_downdog.png" width="500" />
    <img src="images/avg_pixel_intensity_mountain.png" width="500" /> 
</p>

The downdog image looks a bit like a two-humped camel, but the general shape seems pretty decent. Mountain pose makes me feel a little uneasy with the slight creepyness, but it seems very clear and will hopefully do well in the model. 

Next, I created histograms for the Frequency of the pixel intensities for each model. This shows how light and dark each image is, and how defined the shapes are: 

<p float="middle">
    <img src="images/avg_histogram_downdog.png" width="500" />
    <img src="images/avg_histogram_mountain.png" width="500" /> 
</p>


The downdog histogram has a lot more grey area, which you can see in the pixel intensity graphs above, which I expect will make it a bit harder to classify when applied to more than just the mountain/downdog comparison. Mountain is well defined and dark which again, I expect will do well in a model. 

What about edge detection? Since we're working with shapes, maybe looking at the edges of each pose as a feature will do better in the model.

Sobel Filter:

<p float="middle">
    <img src="images/avg_sobel_downdog.png" width="500" />
    <img src="images/avg_sobel_mountain.png" width="500" /> 
</p>


Canny Filter: 

<p float="middle">
    <img src="images/avg_canny_downdog.png" width="500" />
    <img src="images/avg_canny_mountain.png" width="500" /> 
</p>

