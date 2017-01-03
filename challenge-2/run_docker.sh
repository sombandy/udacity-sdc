#!/bin/sh

ip=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')

if [ -z ${ip} ]; then
    echo "ip is not set. Try changing en0 to en1 in the script"
    exit
fi

xhost + $ip

export DISPLAY=localost:0

docker run -it --rm \
    -p 2022:22 \
    --name rosbag_reader \
    --env="DISPLAY=$ip:0" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$HOME/sharefolder:/sharefolder" \
    udacity-sdc  /bin/bash
