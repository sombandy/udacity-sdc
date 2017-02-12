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


### Files Submitted & Code Quality
#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:


1. [model.py](./mode.py) is the script to create and train the model
2. [preprocess.py](./preprocess.py) to preprocess the images before feeding those into the network
3. [drive.py](./drive.py) for driving the car in autonomous mode
4. [model.h5](./model.h5) is a trained keras model of convolution neural network that can drive the car in track1 as well as track2
5. [writeup.md](./writeup.md) is the write up summarizing the approach and the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track1 and track2 by executing
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable
The [model.py](./model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model.

The [preprocess.py](./preprocess.py) file contains the preprocessing code applied to an image before feeding the image into the network.

### Model Architecture and Training Strategy


#### 1. An appropriate model architecture has been employed
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

**Image Augmentation** To reduce the effect of overfitting we increase the number of training data point. Particularly include some noisy training data, e.g. angle shifted images taken from left and right cameras.

#### 4. Appropriate training data

##### Udacity Data
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

Total number of data points in this dataset is **738**

Compare this with the histogram of the steering angle for the Udacity dataset. In this dataset, almost 50% of the data points have close to -1.0 or 1.0 steering angles.

#### Train Validation Split
Through out this project will use 90% - 10% split of the dataset and use the 90% split to train the model and 10% as the validation set. We only keep the center images in the validation set even when we augment the left and right images in the training set.
The validation set will be used to detect overfitting and tune parameters like the number of Epochs for training etc.
The final testing is done on the actual simulator based on whether or not the car is able to drive around in a given track.

###  Architecture and Training Documentation
#### 1. Solution Design Approach
As mentioned at the beginning that we have following goals for the trained model
	1. The model should be able to successfully drive around the car in track1
	2. The model should be able to successfully drive around the car in track2
	3. The model is trained only using the track1 data but it should be able to generalize to track2
	
We will explore following solutions in sequence to achieve our goals
	1. **Baseline** trained only on the center images of the Udacity data
	2. **Image Normalization** Same as baseline, but the images will be cropped and normalized
	3. **Augment Left/Right Images** Left/Right images are augmented after shifting the angle
	4. **Include Sharp Turn Data** In addition to the udacity data we will use the sharp turn dataset that we have created.

#### 1. Baseline

**Training Data**
	- Only the center images of the udacity training data is used
	- There 7232 center images in the training set and 803 images in the validation set

**Preprocessing**
	- The images are resized to *200x66*

**Training Parameters**
We will use the following parameter configuration for all the experiements
	- BATCH_SIZE = 64
	- Optimizer = Adam
	- Learning Rate = 0.0001

**Loss**
![](report_images/baseline_loss.png)

**Observations**
	- Loss does not decrease over iterations. We have not normalized the image pixels. The pixel values range from 0 to 255. So the input to the network is actually quite high value integers. The *tanh* output will be close to -1.0 or 1.0 where the derivative is close to zero. So in each iterations the parameters will be changed by a very small amount. That's why we see a very slow convergence.
	- The loss is quite high. Particularly, the training loss is close to 0.010 after 30 epochs. We will soon see 4x reduction in training loss.

**Final Result** 
	- The car goes out of the road within few seconds of driving. It is not able to take any turn.
 
#### 2. Image Normalization
 In this experiment, we will crop the image and apply normalization
 
**Training Data** Same as above

**Preprocessing**
	1. Cropping: The upper part of the image shows the sky and trees, not relevant to driving decisions. Few rows at the bottom of the image mostly show the bonnet of the car. So we crop the images from 56 to 150 rows of an image
```
img_crop = img[56:150, :, :]
```
The picture below shows the original and the cropped images.
![](report_images/cropping.png)
2. Normalization: We apply per-channel normalization by subtracting the mean and dividing by standard deviation of each channel. This will make most pixel values in the range of 0 and 1, addressing the problem of large input values we saw in case of baseline.
```
def normalize_image(img):
    means = np.mean(img, axis=(0, 1))
    means = means[None,:]
    std = np.std(img, axis=(0, 1))
    std = std[None,:]
    return (img - means) / std
```

**Loss**
![](report_images/normalize_loss.png)

**Observations**
	- Loss decreases much faster than Baseline. Within 5 epochs the validation loss is below 0.010
	- The validation loss starts increasing after epochs 16-17 but the training loss continues to decrease. This is a clear sign of **overfitting**

**Final Results**
	- The care has now learnt to take few truns
	- It is able to cross the bridge in the track1 and after that it goes off the road
	- Overall a very good progress from baseline

#### 2. Final Model Architecture
