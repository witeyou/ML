#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
带回调函数的训练模型,能够随时中止训练的进行
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if (logs.get('accuracy') > 0.87):
            print("\nReached 87% accuracy so cancelling training!")
            self.model.stop_training = True


mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5, callbacks=[myCallback()])
print("==complete training==")
