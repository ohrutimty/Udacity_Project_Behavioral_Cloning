import csv
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sklearn

def load_data(directory):
    allImgPaths = []
    allSteerings = []

    with open(directory + '/driving_log.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)

        for row in reader:
            img_center = row[0]
            img_left = row[1]
            img_right = row[2]
            steering_center = float(row[3])

            # create adjusted steering measurements for the side camera images
            correction = 0.2 # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            # add images and angles to data set
            allImgPaths.append(img_center)
            allImgPaths.append(img_left)
            allImgPaths.append(img_right)

            allSteerings.append(steering_center)
            allSteerings.append(steering_left)
            allSteerings.append(steering_right)

    return (allImgPaths, allSteerings)


## Load Data
from sklearn.model_selection import train_test_split

imgPaths, steerings = load_data("data/mydata_track1")
print('Total Images: {}'.format(len(imgPaths)))

## Split data to training and validation
samples = list(zip(imgPaths, steerings))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))


## Create generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imgPath, steering in batch_samples:
                image_ori = cv2.imread(imgPath)
                image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(steering)
                # Add flipping data
                images.append(cv2.flip(image, 1))
                angles.append(steering * -1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

## Model
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout

def nVidiaModel():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))

    model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.summary()

    return model

## Training
model = nVidiaModel()

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_samples),
    validation_data=validation_generator,
    validation_steps=len(validation_samples),
    epochs=5,
    verbose=1)

model.save('model.h5')
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
