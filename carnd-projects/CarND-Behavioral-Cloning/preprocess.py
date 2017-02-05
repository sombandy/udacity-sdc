#!/usr/bin/env python
#
# Date: Feb-05-2017
# Author: somnath.banerjee

import cv2
import numpy as np

def normalize_image(img):
    means = np.mean(img, axis=(0, 1))
    means = means[None,:]

    std = np.std(img, axis=(0, 1))
    std = std[None,:]
    return (img - means) / std

def preprocess_image(img):
    img_crop = img[56:150, :, :]
    img_resize = cv2.resize(img_crop, (200, 66))
    img_normed = normalize_image(img_resize)
    return img_normed
