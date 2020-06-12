import tensorflow as tf
import matplotlib.pyplot as plt

print(tf.__version__)  # 查看当前版本

mnist = tf.keras.datasets.fashion_mnist  # 导入数据集

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

'''
# 用来查看数据
print(training_labels[0])
print(training_images[0])
print("====over====")
plt.imshow(training_images[0])
plt.show()
'''

# 0-1归一化
training_images = training_images / 255.0
test_images = test_images / 255.0
print("normalized")

# 创建网络模型
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(training_images, training_labels, epochs=5)

print("====COMPLETE TRAINING====")

# 测试模型
model.evaluate(test_images, test_labels)

#
classifications = model.predict(test_images)
print(classifications[0])