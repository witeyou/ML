#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# 对输入的形状进行啊调整,否则后面会报错
img_rows = training_images[0].shape[0]
img_cols = training_images[0].shape[1]
training_images = training_images.reshape(training_images.shape[0], img_rows, img_cols, 1)
test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)

# 归一化
training_images = training_images / 255.0
test_images = test_images / 255.0
print("==completed normalize==")

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(training_images, training_labels, epochs=5)

print("====COMPLETE TRAINING====")

# 测试模型
model.evaluate(test_images, test_labels)
