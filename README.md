# pytorch-segment-nst
I'm working vehicle parts detection on videos and I have a lot of lebeled data with bounding boxes and segmentation pixels that I'm looking for ways to use in order to create some artistic videos!
# Transferring style on a per-frame basis : Project overview
* Understanding neural style transfer theory.
* Exploring optimization and feed forward methods on static images before working on segments of images then videos.
* Stabilizing video results. 
* Transferring different styles in different segments of the image.
* Training custom models.

### NST Introduction

This is what transferring the style of Van Gogh starry night painting into a vehicle content image looks like using the optimization method.

![Game Process](https://github.com/aymanemoataz/pytorch-segment-nst/blob/main/results/bmw_starry.jpg)

### NST algorithm

It uses a set of feature maps from the processing hierarchy of CNN nets like VGG-19 in order to transfer style from a style image into a content image. To better grasp this process we need to learn about content and style reconstruction.

## Part I : Optimization method - Static image style transfer

#### Content reconstruction:

![Game Process](https://github.com/aymanemoataz/pytorch-segment-nst/blob/main/results/noisefiat_Trim.gif)


#### Style Reconstruction:

This is the style reconstruction of the starry night painting, you can already see that the reconstruction is not similar to the content reconstruction

![Game Process](https://github.com/aymanemoataz/pytorch-segment-nst/blob/main/results/style_reconstruction_Trim.gif)






## Code and Resources Used 

**Aleksa's github repo:** https://github.com/gordicaleksa/pytorch-neural-style-transfer

**Pytorch**

**Python Version:** 3.9 

**For requirements:**  ```pip install -r requirements.txt```   

