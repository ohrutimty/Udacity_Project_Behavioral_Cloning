#**Behavioral Cloning**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_cnn_architecture.png "Model Visualization"
[image2]: ./examples/center_sample.jpg "Center sample"
[image3]: ./examples/center_recover_left.jpg "Recovery Left Image"
[image4]: ./examples/center_recover_right.jpg "Recovery Right Image"
[image5]: ./examples/center_sample_flip.jpg "Flipped Image"
[image6]: ./examples/center_sample_track_2.jpg "Track2 sample"
[image7]: ./examples/crop_image.jpg "crop sample"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results
* video_track1.mp4 the car driving autonomously around the track1(Lake)
* video_track2.mp4 the car driving autonomously around the track1(Mountain)

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

Using nvidia "[End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)" model to train the dataset. And add one keras lambda layer and one keras cropping layer.
This is keras model summary:

```sh
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 981,819.0
Trainable params: 981,819.0
Non-trainable params: 0.0
_________________________________________________________________
```

####2. Attempts to reduce overfitting in the model

According to nVidia model, it runs without using any dropout layer or pooling layer. After test this model, the vehicle could keep stay on the track. I have tried to add dropout layer between each fully-connected layer, and there is no problem in track one; however, track 2 cannot pass the first sharp turn. Therefore, I did not add any dropout layer and keep the original model.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

At the beginning of the test, I only use center camera images as my training data. However, the performance is not good, the car runs out of road. Then I combined center, left and right images to train the model, and it can make the car stay on track.

This is the list of data I collected:

-- Track1
- 2 laps of center lane driving
- 1 lap of recovery driving from the sides
- 1 lap of counter-clockwise driving

-- Track2
- 2 laps of dring on right-hand side

Combined all of image to train the model. Add track 2 data is to help the model will generalize better and hope that this model can run on both tracks.
Total images I collected are 39009 samples, and the dataset is split with 80% of training data and 20% of validation data.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

According to udacity course recommendation, I directly use the model provided by nVidia. Besides, the course notes also tell how to collect data, so I follow those steps to get all the data ad mentioned above.

####2. Final Model Architecture

The final model architecture is based on nVidia provides a model of autonomous car, with 5 convolutional layer and 3 fully-connected layer. When input data, it also includes one layer for data normalization by using keras lambda layer. Moreover, this time we do not need whole image to train the model, we only need a part of image to train, so we add one keras cropping layer. In order to output one result for steering, add one fully-connected layer with one neuron to get the inference result.

Here is a visualization of the architecture provided by nVidia:

![alt text][image1]

Lambda layer:
```sh
Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3))
```
Cropping layer:
```sh
Cropping2D(cropping=((50,20), (0,0)))
```

The epoch I use for training is 5 and it seems like being enough for training.

####3. Creation of the Training Set & Training Process

About the simulator resolution I choose is 640*480 and the graphic quality is fastest, the output image resolution 320\*160.

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to get back to the center of the track when the vehicle is deviated from the center. These images show when the car is close to left and right.

![alt text][image3]
![alt text][image4]

Then I also add one counter-clockwise track to balance the dataset and generalize dataset.

To augment the data, I also flipped images and angles in the generator. For example, here is an image that has then been flipped:

![alt text][image2]
![alt text][image5]

Moreover, in order to make the model is able to run not only on track1 but in track2, I add two laps of driving on track2 in dataset. Because there is a white dotted line on the center of the road, thus I try my best to keep the car on right-hand side. But it's not easy to do, in autonomous mode it is usually run to the left hand side when the car is turning.

![alt text][image6]

In order to make the model can be focused on the road, so we remove useless information in the image by cropping the image in keras cropping layer. Delete 50 rows pixels from the top of the image and 20 rows pixels from the bottom of the image. Here is an example:

![alt text][image7]
