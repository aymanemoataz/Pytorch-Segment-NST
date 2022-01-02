# pytorch-segment-nst
I am working at the moment on vehicle parts detection on videos and I have a lot of labeled data with bounding boxes and segmentation pixels that I am looking for ways to use in order to create some artistic videos!
# Transferring style on a per-frame basis : Project overview
* Understanding neural style transfer theory.
* Exploring optimization and feed forward methods on static images before working on segments of images then videos.
* Stabilizing video results. 
* Transferring different styles in different segments of the image.
* Training custom models.

## What I am currently working on ?

I recently learned about neural style transfer and know I am really curious to combine it with semantic segmentation on car videos. The naive video results are not stable and we need to explore some techniques to stabilize them. You can see below video results for a stable car video

<p align="center">

<img src="https://github.com/aymanemoataz/pytorch-segment-nst/blob/main/results/composed_background_masked_1_Trim.gif" width="400px" height="300px">
<img src="https://github.com/aymanemoataz/pytorch-segment-nst/blob/main/results/car1_Trim.gif" width="400px" height="300px">
</p>

### Neural style transfer

This is what transferring the style of Van Gogh's starry night painting into a vehicle content image looks like using the optimization method.


<p align="center">

<img src="https://github.com/aymanemoataz/pytorch-segment-nst/blob/main/results/bmw_starry.jpg" width="400px" height="300px">
<img src="image_nst/data/style-images/vg_starry_night_resized.jpg" width="267px" height="300px">
</p>





### NST algorithm

It uses a set of feature maps from the processing hierarchy of CNN nets like VGG-19 in order to transfer style from a style image into a content image. 
## Part I : Optimization method - Static image style transfer

This is a small visualization of the quick initial change of the content loss and the style loss during the neural style transfer process (using the optimizer method). To better grasp this process we need to learn about content and style reconstruction. (We're using the night café painting as the style image here)

<p align="center">
<img src="https://github.com/aymanemoataz/pytorch-segment-nst/blob/main/results/NST_Optimizer_Trim.gif" width="450px" height="300px">
</p>

<p align="center">
<img src="image_nst/data/style-images/vg_la_cafe.jpg" width="267px" height="300px">
<img src="https://github.com/aymanemoataz/pytorch-segment-nst/blob/main/results/fiatlacafe.jpg" width="400px" height="300px">
</p>


#### Content reconstruction:
We first visualize the content reconstruction of a vehicle image from a noise image (the reconstruction of content is very fast using the L-BFGS optimizer compared to the Adam optimizer but can have a slow run-time especially without a gpu), we used a tesla p 100 with 16gb(available with colab pro) to compute the results below in under 14 seconds.

![Game Process](https://github.com/aymanemoataz/pytorch-segment-nst/blob/main/results/noisefiat_Trim.gif)


#### Style Reconstruction:
This is the style reconstruction of the starry night painting, you can see that contrary to content reconstruction, the content of the style image is not preserved. The output image has the same style of the starry night style image. 

![Game Process](https://github.com/aymanemoataz/pytorch-segment-nst/blob/main/results/style_reconstruction_Trim.gif)






## Code and Resources Used 

**Aleksa's github repo:** https://github.com/gordicaleksa/pytorch-neural-style-transfer

**Pytorch**

**Python Version:** 3.9 


