#!/usr/bin/env python
#
# Date: Feb-05-2017
# Author: somnath.banerjee

import datetime
import json
import keras
import matplotlib.pylab as plt
import models
import numpy as np
import os
import preprocess
import sys
import random
import time

class DataHelper:
    def __init__(self, data_file):
        xs = []
        ys = []

        with open(data_file) as f:
            header = f.readline()
            dirname = os.path.dirname(data_file)
            for line in f:
                fields = line.split(", ")
                # Loading only the center images
                xs.append(os.path.join(dirname, fields[0]))
                ys.append(np.float32(fields[3]))

        c = list(zip(xs, ys))
        random.shuffle(c)
        xs, ys = zip(*c)

        self._batch_pointer = 0
        self._train_xs = xs[:(int)(len(xs) * 0.9)]
        self._train_ys = ys[:(int)(len(xs) * 0.9)]

        val_xs = []
        val_ys = []
        val_size = (int)(len(xs) * 0.1)
        for i in range(val_size):
            img = plt.imread(xs[-i])
            img_pre = preprocess.preprocess_image(img)

            val_xs.append(img_pre)
            val_ys.append(ys[-i])

        self._val_xs = np.asarray(val_xs)
        self._val_ys = np.asarray(val_ys)

    def data_size(self):
        return len(self._train_ys) + len(self._val_ys)

    def val_data(self):
        return self._val_xs, self._val_ys

    def next_train_batch(self, batch_size):
        x_out = []
        y_out = []
        for i in range(batch_size):
            data_idx = (self._batch_pointer + i) % len(self._train_ys)

            img = plt.imread(self._train_xs[data_idx])
            img_pre = preprocess.preprocess_image(img)

            x_out.append(img_pre)
            y_out.append(self._train_ys[data_idx])

        self._batch_pointer += batch_size
        return np.asarray(x_out), np.asarray(y_out)

def train(config):
    dh = DataHelper(config['data_file'])
    data_size = dh.data_size()
    val_x, val_y = dh.val_data()

    print("Number of training data", data_size, ". data shape", val_x.shape)

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    evaluate_at = config["evaluate_at"]

    step_start = time.time()
    num_steps = epochs * data_size // batch_size

    model = models.nvidia_end2end()
    print(model.summary())

    print("Training for %d steps" % num_steps)
    for steps in range(num_steps):
        x, y = dh.next_train_batch(batch_size)
        train_loss = model.train_on_batch(x, y)

        if steps % evaluate_at == 0:
            time_taken = time.time() - step_start
            print("Epoch {} Steps {} time taken {:0.1f}s".format(
                steps * batch_size // data_size, steps, time_taken))

            detailed_loss(model, x, y, "Train")
            detailed_loss(model, val_x, val_y, "Validation")

            step_start = time.time()
            model.save(config["model_file"])

    model.save(config["model_file"])
    sample_output(model, val_x, val_y)

def detailed_loss(model, x, y, dataname):
    fwd_idx = (y >= -1e-6) & (y <= 1e-6)
    left_idx = y < -1e-6
    right_idx = y > 1e-6

    overall = model.evaluate(x, y, verbose=0)
    forward = model.evaluate(x[fwd_idx, ], y[fwd_idx], verbose=0)
    left = model.evaluate(x[left_idx, ], y[left_idx], verbose=0)
    right = model.evaluate(x[right_idx, ], y[right_idx], verbose=0)

    print("{} loss\t {:0.6f} forward {:0.6f} left trun {:0.6f} right turn "
          "{:0.6f}".format(dataname, overall, forward, left, right))

def sample_output(model, x, y, K=10):
    angle_type = ["Forward", "Left", "Right"]
    angles = [(y >= -1e-6) & (y <= 1e-6), y < -0.01, y > 0.01]

    for i in range(3):
        print("Steering angle: " + angle_type[i])
        _y = y[angles[i]]
        _x = x[angles[i], ]

        test_idx = np.random.randint(0, len(_y), size=K)
        y_pred = model.predict(_x[test_idx, ]).reshape(K)
        diff = y_pred - _y[test_idx, ]

        print("Actual\t\tPredicted\tDiff")
        for i in zip(_y[test_idx,], y_pred, diff):
            print("{:0.6f}\t{:0.6f}\t{:0.6f}".format(*i))

def main():
    if len(sys.argv) != 2:
        print("Usage %s traing_config_file" % sys.argv[0])
        return

    config = {}
    with open(sys.argv[1]) as f:
        config = json.load(f)

    train_start = time.time()
    train(config)
    print("Total training time %s" %
          str(datetime.timedelta(seconds=int(time.time() - train_start))))

if __name__ == "__main__":
    main()
