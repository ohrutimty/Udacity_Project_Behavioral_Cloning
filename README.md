# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files:
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :).

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

[//]: # (Image References)

[image1]: ./examples/nvidia_cnn_architecture.png "Model Visualization"
[image2]: ./examples/center_sample.jpg "Center sample"
[image3]: ./examples/center_recover_left.jpg "Recovery Left Image"
[image4]: ./examples/center_recover_right.jpg "Recovery Right Image"
[image5]: ./examples/center_sample_flip.jpg "Flipped Image"
[image6]: ./examples/center_sample_track_2.jpg "Track2 sample"
[image7]: ./examples/crop_image.jpg "crop sample"
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
