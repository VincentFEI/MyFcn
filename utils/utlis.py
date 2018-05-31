import tensorflow as tf
import numpy as np
from PIL import Image
import os
import re
import sys
import time
import random


def sparse_softmax_cross_entropy_ignore_labels(logits=None, labels=None, name=None, ignores=None):
    num_classes = logits.get_shape()[3]
    labels_onehot = tf.one_hot(labels, num_classes)

    if ignores is None:
        ignores = num_classes - 1

    t1 = tf.zeros_like(labels_onehot[:, :, :, :ignores])
    t2 = tf.expand_dims(labels_onehot[:, :, :, ignores], -1)
    t3 = tf.zeros_like(labels_onehot[:, :, :, ignores + 1:])
    t4 = tf.concat(axis=3, values=[t1, t2, t3])
    logits_fix = tf.where(t4 > 0, 1e30 * tf.ones_like(logits), logits)

    if name is None:
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits_fix, labels=labels_onehot)
    else:
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits_fix, labels=labels_onehot, name=name)

    return loss

# def random_crop(image, annotation, height, width):
#     old_width = image.shape[1]
#     old_height = image.shape[0]
#     assert(old_width >= width)
#     assert(old_height >= height)
#     max_x = max(old_height-height, 0)
#     max_y = max(old_width-width, 0)
#     offset_x = random.randint(0, max_x)
#     offset_y = random.randint(0, max_y)
#     image = image[offset_x:offset_x+height, offset_y:offset_y+width]
#     annotation = annotation[offset_x:offset_x+height, offset_y:offset_y+width]
#
#     assert(image.shape[0] == height)
#     assert(image.shape[1] == width)
#
#     return image, annotation
#
#
# def random_crop_soft(image, annotation, max_crop):
#     offset_x = random.randint(1, max_crop)
#     offset_y = random.randint(1, max_crop)
#
#     if random.random() > 0.5:
#         image = image[offset_x:, offset_y:, :]
#         annotation = annotation[offset_x:, offset_y:, :]
#     else:
#         image = image[:-offset_x, :-offset_y, :]
#         annotation = annotation[:-offset_x, :-offset_y, :]
#
#     return image, annotation


# from utils import ReadDataset
# if __name__ == '__main__':
#     train_image_path = "/home/vincentfei/PycharmProjects/dataset/leftImg8bit/train"
#     train_label_path = "/home/vincentfei/PycharmProjects/dataset/gtFine/train"
#     # train_dataset = ReadDataset.ReadDataset(image_path=train_image_path, label_path=train_label_path)
#
#     train_dataset = ReadDataset.ReadDataset(image_path=train_image_path)
#     start = time.time()
#     batch = train_dataset.get_next_batch(2)
#     end = time.time()
#     print("Times cost : ", float(end-start))
#     print(batch[0].shape)
#     batch = train_dataset.get_next_batch(2)
#     batch = train_dataset.get_next_batch(2)
#     batch = train_dataset.get_next_batch(2)



