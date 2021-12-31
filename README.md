# pytorch-segment-nst
I'm working vehicle parts detection on videos and I have a lot of lebeled data with bounding boxes and segmentation pixels that I'm looking for ways to use in order to create some artistic videos!
# Transferring style on a per-frame basis : Project overview
* Understanding neural style transfer theory.
* Exploring optimization and feed forward methods on static images before working on segments of images then videos.
* Stabilizing video results. 
* Transferring different styles in different segments of the image.
* Training custom models.


## Part I : Optimization method - Static image style transfer
This is what transferring the style of van gogh starry night image into a vehicle image looks like using the optimization method
![Game Process](https://github.com/aymanemoataz/pytorch-segment-nst/blob/main/results/noisefiat.jpg)
### How does it work?

#### Content reconstruction:
![Game Process](https://github.com/aymanemoataz/pytorch-segment-nst/blob/main/results/NST_Optimizer.gif)


#### Style Reconstruction:
This is the style reconstruction of the starry night painting, you can already see that the reconstruction is not similar to the content reconstruction
![Game Process](https://github.com/aymanemoataz/pytorch-segment-nst/blob/main/results/style_reconstruction.gif)






## Code and Resources Used 

**Aleksa's github repo:** https://github.com/gordicaleksa/pytorch-neural-style-transfer

**Pytorch**

**Python Version:** 3.9 

**For requirements:**  ```pip install -r requirements.txt```   
