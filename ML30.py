#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pylab as plt
import numpy as np
from scipy import misc

# misc包是一个包含许多示例图像的包
# ascent是其中的一幅图像
i = misc.ascent()

i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]

# 翻译:不同的过滤器有不同的效果,诸如增强直线以及锐利的边缘等
# 卷积核的不同会带来明显不同的效果
# 通常来说,卷积核中的所有数字加起来应该给是0或者1.如果不是这个数值,则需要额外的设置权重系数"weight"
# 这个给权重系数一般是选择"1"来进行规范化.
# This filter_my detects edges nicely
# It creates a convolution that only passes through sharp edges and straight lines.
# Experiment with different values for fun effects.
# filter_my = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]]
# A couple more filters to try for fun!
# 这个容易保留垂直特征
filter_my = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
# 这个容易保持水平特征
# filter_my = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
# If all the digits in the filter_my don't add up to 0 or 1, you
# should probably do a weight to get it to do so
# so, for example, if your weights are 1,1,1 1,2,1 1,1,1
# They add up to 10, so you would set a weight of .1 if you want to normalize them
weight = 1

# 卷积操作
for x in range(1, size_x - 1):
    for y in range(1, size_y - 1):
        output_pixel = 0.0
        output_pixel = output_pixel + (i[x - 1, y - 1] * filter_my[0][0])
        output_pixel = output_pixel + (i[x, y - 1] * filter_my[0][1])
        output_pixel = output_pixel + (i[x + 1, y - 1] * filter_my[0][2])
        output_pixel = output_pixel + (i[x - 1, y] * filter_my[1][0])
        output_pixel = output_pixel + (i[x, y] * filter_my[1][1])
        output_pixel = output_pixel + (i[x + 1, y] * filter_my[1][2])
        output_pixel = output_pixel + (i[x - 1, y + 1] * filter_my[2][0])
        output_pixel = output_pixel + (i[x, y + 1] * filter_my[2][1])
        output_pixel = output_pixel + (i[x + 1, y + 1] * filter_my[2][2])
        output_pixel = output_pixel * weight
        if output_pixel < 0:
            output_pixel = 0
        if output_pixel > 255:
            output_pixel = 255
        i_transformed[x, y] = output_pixel

# 最大池化
new_x = int(size_x/2)
new_y = int(size_y/2)
newImage = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
  for y in range(0, size_y, 2):
    pixels = []
    pixels.append(i_transformed[x, y])
    pixels.append(i_transformed[x+1, y])
    pixels.append(i_transformed[x, y+1])
    pixels.append(i_transformed[x+1, y+1])
    pixels.sort(reverse=True)
    newImage[int(x/2),int(y/2)] = pixels[0]

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.imshow(i_transformed)
plt.imshow(newImage)
plt.show()
