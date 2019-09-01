#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:44:42 2019

@author: ymguo

#code1.Finish 2D convolution/filtering by your self. 

What you are supposed to do can be described as "median blur", which means by using a sliding window 
on an image, your task is not going to do a normal convolution, but to find the median value within 
that crop.

You can assume your input has only one channel. (a.k.a a normal 2D list/vector)
And you do need to consider the padding method and size. There are 2 padding ways: REPLICA & ZERO. When 
"REPLICA" is given to you, the padded pixels are same with the border pixels. E.g is [1 2 3] is your
image, the padded version will be [(...1 1) 1 2 3 (3 3...)] where how many 1 & 3 in the parenthesis 
depends on your padding size. When "ZERO", the padded version will be [(...0 0) 1 2 3 (0 0...)]

Assume your input's size of the image is W x H, kernel size's m x n. You may first complete a version 
with O(W·H·m·n log(m·n)) to O(W·H·m·n·m·n)).

Follow up 1: Can it be completed in a shorter time complexity?

注：'median blur'作业的考点是找中值的算法（如何降低时间复杂度）。

"""    

'''
example:
def medianBlur(img, kernel, padding_way):
    img & kernel is List of List; padding_way a string
    Please finish your code under this blank
'''
# reference： https://files-cdn.cnblogs.com/files/Imageshop/MedianFilterinConstantTime.pdf

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

def medianBlur(img, radius, padding_way):
    if isinstance(img, list):
        img = np.array(img, dtype=np.uint8)
    MB = MedianBlur(image=img, r=radius, padding_way=padding_way)
    MB.get_median_image()
    return MB.median_blur_image

'''
类实现
'''
class MedianBlur():
    """
    Median Blurring in O(1) Runtime Complexity.
    """
    def __init__(self, image, r, padding_way):
        self.image = image
        self.r = r  # kernel半径
        self.padding_way = padding_way
        self.H, self.W = image.shape
        self.img = self.get_padding_img()
        self.median_blur_image = np.zeros(image.shape, dtype=np.uint8)
        """
        初始化第一行:
        """
        """ (1) 创建空矩阵: """
        # 记录包括padding在内的每一列;
        # 统计的每个小column只有2*r+1维, 为了可加, 同一到256维:
        self.h_hist = np.zeros((256, self.W+2*r), dtype=np.uint8)
        # 记录当前kernel所在位置的每一列:
        self.H_hist = np.zeros((256, 2*r+1), dtype=np.uint8)
        """ (2) 初始化h_hist: """
        for j in range(self.W + 2 * r):
            self.h_hist[:, j] = self.get_hist(j)

    def get_median_image(self):
        r = self.r
        # (i, j) is the position of current central pixel in image.
        for i in range(self.H):
            for j in range(self.W):
                i_img = i + r  # current i corresponding to i+r in padded-image img
                j_h = j + r  # current j corresponding to j+r in h_hist
                """ 1.Update h_hist and H_hist: """
                if i == 0:  # first line processed separately
                    self.H_hist = self.h_hist[:, j_h - r: j_h + r + 1]
                else:
                    if j == 0:  # first element of each line processed separately
                        j_move_down = list(range(j_h - r, j_h + r + 1))  # (2*r+1) columns in total.
                        self.move_down(i_img, j_move_down)  # first (2*r+1) columns of h_hist updated.
                        self.H_hist = self.h_hist[:, j_h - r: j_h + r + 1]  # H_hist updated.
                    else:
                        j_move_down = j_h + r
                        self.move_down(i_img, j_move_down)  # next one column in h_hist updated.
                        self.move_right(j_h)  # H_hist updated.

                """ 2.Use updated H_hist to get median value to update median_image: """
                self.median_blur_image[i, j] = self.get_median()

    
    """
    0. 定义padding函数
    """
    #np.pad?
    # np.pad(array, pad_width, mode, **kwargs)
    #>>> a = [1, 2, 3, 4, 5]
    #>>> np.pad(a, (2,3), 'constant', constant_values=(4, 6))
    #array([4, 4, 1, 2, 3, 4, 5, 6, 6, 6])
    #>>> np.pad(a, (2, 3), 'edge')
    #array([1, 1, 1, 2, 3, 4, 5, 5, 5, 5])
    def get_padding_img(self):
        assert self.padding_way in ["REPLICA", "ZERO"]
        if self.padding_way == "ZERO":
            img = np.pad(self.image, pad_width=self.r, mode='constant', constant_values=0)
        else:
            img = np.pad(self.image, pad_width=self.r, mode='edge')
        return img

    """
    1. 定义get_hist函数
    """
    def get_hist(self, j):
        col = self.img[0:(2 * self.r + 1), j]
        hist = np.zeros(256, dtype=np.uint8)
        for val in col:
            hist[val] += 1

        return hist

    """
    2. 定义位移函数: move_down(), move_right()
    The two steps of the Constant Time Median Filtering algorithm.
    """
    # First
    # update h_hist:
    def move_down(self, i, j_move):
        # i is the line of current central pixel in padded-image img;
        # j_move is the list of columns to be moved in h_hist.
        i_remove = i - self.r - 1
        i_add = i + self.r
        if isinstance(j_move, int):
            j_move = [j_move, ]
        for j in j_move:
            remove_val = self.img[i_remove, j]
            add_val = self.img[i_add, j]
            self.h_hist[remove_val, j] -= 1
            self.h_hist[add_val, j] += 1

    # Second
    # update H_hist:
    def move_right(self, j_h):
        # j_h is the column of current central pixel in h_hist.
        # 注意: 此时的H_hist尚未更新,还处于上一个位置;但j已经更新,是当前位置的列.
        # j_remove = j_h - r - 1是相对于h_hist来说的,事实上H_hist永远只需要del第0列.
        j_add = j_h + self.r
        self.H_hist = np.c_[np.delete(self.H_hist, 0, axis=1),
                            self.h_hist[:, j_add]]
        # np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。np.r_是按列连接。

    """
    3. 定义求中值函数: get_median()
    """
    def get_median(self):
        # get median value for current H_hist
        hist = np.sum(self.H_hist, axis=1)
        #np.sum?
        #>>> np.sum([[0, 1], [0, 5]], axis=0)
        #array([0, 6]) 按列相加
        #>>> np.sum([[0, 1], [0, 5]], axis=1)
        #array([1, 5]) 按行相加
        thres = (2 * self.r + 1) ** 2 // 2 + 1
        sum_cnt = 0
        median = 0
        for val in range(256):
            cnt = hist[val]
            sum_cnt += cnt
            if sum_cnt >= thres:
                median = val
                break
        return median

'''
Test
'''
image = cv2.imread('20190712182540.jpg')
#image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
img = cv2.resize(image, dsize=(int(image.shape[1]/2), int(image.shape[0]/2)))
#plt.imshow(img)

#cv2.imshow('before', img)
#cv2.waitKey(0) # 按任意键关闭显示图片
#cv2.destroyAllWindows()
#cv2.waitKey(1)
#cv2.waitKey(1)
#cv2.waitKey(1)
#cv2.waitKey(1)

H, W, C = img.shape
print(H, W, C)
B, G, R = cv2.split(img)

# Radius: r = 1
# kernel size = 2 * r + 1 = 3
r = 1
padding_way="REPLICA"

bgr_3 = list(map(lambda _: medianBlur(_, r, padding_way), 
                 [B, G, R]))
mb_img_3 = cv2.merge(bgr_3)

# Radius: r = 2
# kernel size = 2 * r + 1 = 5
mb_img_5 = cv2.merge(list(map(lambda _: medianBlur(_, 2, padding_way), [B, G, R])))
# Radius: r = 3
# kernel size = 2 * r + 1 = 7
mb_img_7 = cv2.merge(list(map(lambda _: medianBlur(_, 3, padding_way), [B, G, R])))


#cv2.imshow('original', img)
#cv2.imshow('test01', mb_img_3)
#cv2.imshow('test02', mb_img_5)
#cv2.imshow('test03', mb_img_7)
#
#cv2.waitKey(0) # 按任意键关闭显示图片
#cv2.destroyAllWindows()
#cv2.waitKey(1)
#cv2.waitKey(1)
#cv2.waitKey(1)
#cv2.waitKey(1)

'''
比较一下不同kernel size的处理效果:
    并存储在result文件夹中
'''
save_path = "./result_medianBlur"
if not os.path.exists(save_path):
    os.mkdir(save_path)
    
plt.figure(figsize=(15, 10))
plt.subplot(221)
plt.title('original')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(222)
plt.title('kernel size = 3')
plt.imshow(cv2.cvtColor(mb_img_3, cv2.COLOR_BGR2RGB))
plt.subplot(223)
plt.title('kernel size = 5')
plt.imshow(cv2.cvtColor(mb_img_5, cv2.COLOR_BGR2RGB))
plt.subplot(224)
plt.title('kernel size = 7')
plt.imshow(cv2.cvtColor(mb_img_7, cv2.COLOR_BGR2RGB))

plt.savefig(save_path+"/medianBlur_kernels.jpg", dpi=1000)
plt.show()



'''
对比Opencv的快速效果：
'''
cv2_mb_img_7 = cv2.medianBlur(img, ksize=7)  # uses #BORDER_REPLICATE

cv2.imshow('test03_cv2',cv2_mb_img_7)
cv2.waitKey(0) # 按任意键关闭显示图片
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)




















