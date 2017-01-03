# Baseline setup for Udacity SDC Challenge 2

### Prerequisites
- MAC OSX (Not tested but should work on Linux as well)
- Docker installed and running

### Install XQuartz on MAC
This step is not necessary for Linux machines
```
brew cask install xquartz
```
Or install from https://www.xquartz.org/.

To make sure GUI is working from docker you can follow the steps mentioned in [Docker for Mac and GUI applications](https://fredrikaverpil.github.io/2016/07/31/docker-for-mac-and-gui-applications/)


### Setup Sully Chen's Nvidia Autopliot code and data

Create a shared folder to be used inside docker and download the code and data there.
- mkdir ~/sharefolder
- cd ~/sharefolder
- git clone https://github.com/SullyChen/Autopilot-TensorFlow.git
- cd Autopilot-TensorFlow
- Download and unzip the [driving_dataset](https://drive.google.com/file/d/0B-KJCaaF7ellQUkzdkpsQkloenM/view)

### Build the docker image
```
./build_docker.sh
```

### Run the docker image
```
./run_docker.sh
```

### Train the model
```
# Inside the docker
cd /sharefolder/Autopilot-TensorFlow
python train.py
```
You can stop the above after 1000 steps or so just to see everything is working.

### Now time to see the machine is steering a car
```
python run_dataset.py
```

Have fun. Thanks to [Sully Chen](https://github.com/SullyChen) for making the code public. I just put together the setup instructions that worked for me.
