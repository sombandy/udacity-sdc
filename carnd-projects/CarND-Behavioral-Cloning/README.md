## Behavioral Cloning Project
The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior
- Build, a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report

We kept 2 additional goals

1. Train the model only using track1 data. Never show the model data of track2. Yet the car should be able to complete track2
2. The car should be able to drive upto decent length in presence of shadow as well. Driving around in track2 in presence of shadow is *not* a goal

### Files Submitted & Code Quality
#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:


1. model.pycontaining the script to create and train the model
2. preprocess.py contains the code to preprocess the images before feeding those in the network
3. drive.py for driving the car in autonomous mode
4. [model.h5](./model.h5) containing a trained convolution neural network
5. REDME.md containing the wirte up summarizing the approach and the results

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```
python drive.py model.h5
```

#### 3. Submssion code is usable and readable
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The preprocess.py file contains the preprocessing code applied to an image before feeding the image into the network.

### Model Architecture and Training Strategy


#### 1. An appropriate model arcthiecture has been employed
I have used the same model used in [Nvidia's End To End driving](https://arxiv.org/pdf/1604.07316.pdf). The model takes a 200 x 66 image and predicts the desired steering angle of the car.

The model layers and parameters are as follows

Layer (type) | Filter Shape | Subsample | Output Shape | Activation | Param #
--- | --- | --- | --- | --- |  ---
input | None | None | 66 x 200 x3 | rele | 0
convolution 1 | 24@5x5 | 2x2 | 31 x 98 x 24  | relu | 1,824
convolution 2 | 36@5x5 | 2x2 | 14 x 47 x 26 | relu | 21,636
convolution 3 | 48@5x5 | 2x2 | 5 x 22 x 48 | relu | 43,248
convolution 4 | 64@3x3 | None | 3 x 20 x 64 | relu | 27,712
convolution 5 | 64@3x3 | None | 1 x 18 x 64 | relu | 36,928
Flatten | None | None | 1152 | relu | 0
Fully Connected 1 | None  | None | 1164 | relu | 1,342,092
Fully Connected 2 | None | None | 100 | relu | 116,500
Fully Connected 3 | None | None | 50 | relu |5050
Fully Connected 4 | None | None | 10 | relu |510
Fully Connected 5 | None | None | 1 | tanh | 11

Total number of parameters = 1,595,511

#### 2. Attempts to reduce overfitting in the model
**Dropout**  To reduce overfitting applied dropout after the following layers
1. Flatten
2. Fully Connected 1
3. Fully Connected 2
Through out dropout probablity was set to 0.5

**Early Stopping** To avoid overfitting we adopted early stopping. Typically we stop after 20-30 Epochs

#### 4. Appropriate training data

##### Udacity Sample Training Data
Started with Udacity provided [sample training data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) for track1.

##### Creation of the Training Set & Training Process

Below are the quick summary of the data.
Steering Angle | Number of Data Points | Percentage
---- | --- | ---
0.0  (straight) | 4361 | 54%
Less than 0.0 (right turn) | 1776 | 22%
Greater than 0.0 (left turn) | 1900 | 24%
Total | 8037 | 100%

###  Architecture and Training Documentation
#### 1. Solution Design Approach
As mentioned at the beginning that we have 2 major goals for our solution
	1. Make sure the car is able to drive in track2 while it is trained using the data of track1 only
	2. The car should be able to drive around in presence of shadow as well

We will explore following solutions in sequence
	1. **baseline** trained only on the center images of the Udacity data
	2. **Image Normalization** Same as baseline, but the images will be cropped and normalized
	3. **Augment Left/Right Images** Left/Right images are augmented after shifting the angle
	4. **Include More Training Data**
	5.	**Introduce Shadow and other preprocessing of the images**
	6.
#### 2. Final Model Architecture
