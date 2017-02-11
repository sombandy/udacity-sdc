## Behavioral Cloning Project
The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior
- Build, a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report

We kept 2 additional goals
1. Test that the model successfully drives around track2 without leaving the road
2. Train the model only using track1 data. Never show the model data of track2. Yet the car should be able to successfully drive around track2
2. The car should be able to drive up to decent length in presence of shadow as well. Driving around in track2 in presence of shadow is *not* a goal

### Files Submitted & Code Quality
#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:


1. [model.py](./mode.py) is the script to create and train the model
2. [preprocess.py](./preprocess.py) to preprocess the images before feeding those into the network
3. [drive.py](./drive.py) for driving the car in autonomous mode
4. [model.h5](./model.h5) is a trained keras model of convolution neural network that can drive the car in track1 as well as track2
5. [REDME.md](./README.md) is the write up summarizing the approach and the results

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track1 and track2 by executing
```
python drive.py model.h5
```

#### 3. Submssion code is usable and readable
The [model.py](./model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model.

The [preprocess.py](./preprocess.py) file contains the preprocessing code applied to an image before feeding the image into the network.

### Model Architecture and Training Strategy


#### 1. An appropriate model arcthiecture has been employed
I have used the same model used in [Nvidia's End To End driving](https://arxiv.org/pdf/1604.07316.pdf). The model takes a *200 x 66* image and predicts the desired steering angle of the car.

The model layers and parameters are as follows

Layer (type) | Filter Shape | Subsample | Output Shape | Activation | Param #
--- | --- | --- | --- | --- |  ---
input | - | - | 66 x 200 x3 | relu | 0
convolution 1 | 24@5x5 | 2x2 | 31 x 98 x 24  | relu | 1,824
convolution 2 | 36@5x5 | 2x2 | 14 x 47 x 26 | relu | 21,636
convolution 3 | 48@5x5 | 2x2 | 5 x 22 x 48 | relu | 43,248
convolution 4 | 64@3x3 | None | 3 x 20 x 64 | relu | 27,712
convolution 5 | 64@3x3 | None | 1 x 18 x 64 | relu | 36,928
Flatten | - | - | 1152 | relu | 0
Fully Connected 1 | -  | - | 1164 | relu | 1,342,092
Fully Connected 2 | - | - | 100 | relu | 116,500
Fully Connected 3 | - | - | 50 | relu |5050
Fully Connected 4 | - | - | 10 | relu |510
Fully Connected 5 | - | - | 1 | tanh | 11

Total number of parameters = 1,595,511

Since we are using tanh activation the output is restricted between -1 to +1.

#### 2. Attempts to reduce overfitting in the model
**Dropout**  To reduce overfitting applied dropout after the following layers
1. Flatten
2. Fully Connected 1
3. Fully Connected 2
Through out dropout probablity was set to 0.5

**Image Augmentation** To reduce the effect of overfitting we increase the number of training data point. Particularly in include some noisy training data, e.g. angle shifted images taken from left and right cameras.

#### 4. Appropriate training data

##### Udacity Sample Training Data
Udacity provided [sample training data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) for track1. We will use the Udacity data as the main source of our training and validation data.

##### Quick summary of the Udacity data
Steering Angle | Number of Data Points | Percentage
---- | --- | ---
0.0  (straight) | 4361 | 54%
Less than 0.0 (right turn) | 1776 | 22%
Greater than 0.0 (left turn) | 1900 | 24%
Total | 8037 | 100%

The histogram of the steering angles are shown in the figure below
![](report_images/udacity_data_angle_hist.png)

We can see that the Udacity data is heavily skewed towards driving straight at steering angle close to 0.0. This is good enough for track1. But to drive around in the track2 the car needs to take some sharp turns. Udacity data is not enough to train the model in taking sharp turns. 

#### Sharp Turn Data
To train the model to take sharp turns we created some sharp turn data. In particular, we recorded some dataset where the car is *driving away* from the curb or fence. We positioned the car very close to a curb or a fence and then start recording while it takes sharp turn to drive away from the curb/fence. Note that as mentioned in the goals we created this data set from track1 only.

The full video of this training data is available in the link below

[![](https://img.youtube.com/vi/4Twg8Gj2Nuk/0.jpg)](https://youtu.be/4Twg8Gj2Nuk)

An example image of this training dataset from all three; left, center and right cameras are shown below.
![](report_images/sharp_turn_bridge.png)

The histogram of the steering angles for this dataset is shown below.
![](report_images/sharp_turn_angle_hist.png)

Compare this with the histogram of the steering angle for the Udacity dataset. In this dataset almost 50% of the data points have close to -1.0 or 1.0 steering angles.

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
