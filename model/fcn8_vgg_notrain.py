from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf

class FCN8VGG:

    def __init__(self):
        # 读取VGG16参数，并且存入一个字典
        vgg16_npy_path = "../fcn/vgg16.npy"
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        # L2正则化项的权重
        self.wd = 5e-4
        print("npy file loaded")


    def build(self, batch_images, num_classes = 21, keepprob = 1, debug = False, random_init_fc8 = False):

        # 第一级卷积层
        self.conv1_1 = self._conv_layer(batch_images, "conv1_1")
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self._max_pool(self.conv1_2, "pool1", debug)

        # 第二级卷积层
        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self._max_pool(self.conv2_2, 'pool2', debug)

        # 第三级卷积层
        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self._max_pool(self.conv3_3, 'pool3', debug)

        # 第四级卷积层
        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self._max_pool(self.conv4_3, 'pool4', debug)

        # 第五级卷积层
        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self._max_pool(self.conv5_3, 'pool5', debug)

        # 第六级卷积层（VGG网络中的全连接层，但在FCN中作为卷积层）
        # 原本在VGG中这一层的featuremap的尺寸是7*7，输入通道数为512，输出通道数为4096，所有参数的shape就是[25088,4096]
        # 但是这里需要更改成卷积层，因此参数需要reshape成[7,7,512,4096]
        self.fc6 = self._fc_layer(self.pool5, "fc6")
        self.fc6 = tf.nn.dropout(self.fc6, keep_prob=keepprob)

        # 第七级卷积层（VGG网络中的全连接层，但在FCN中作为卷积层）
        self.fc7 = self._fc_layer(self.fc6, "fc7")
        self.fc7 = tf.nn.dropout(self.fc7, keep_prob=keepprob)

        # 第八级卷积层（VGG网络中的全连接层，但在FCN中作为卷积层）
        # 这一级卷积层的输出通道数应该与num_classes相等
        if random_init_fc8:
            self.score_fr = self._score_layer(self.fc7, "score_fr", num_classes)
        else:
            self.score_fr = self._fc_layer(self.fc7, "score_fr", num_classes=num_classes, relu=False)

        self.pred = tf.argmax(self.score_fr, dimension=3)

        # 上采样2倍
        # score_fr层相当于原图的1/32
        # 对score_fr层向上采样，增大了一倍
        self.upscore2 = self._upscore_layer(self.score_fr,
                                            shape=tf.shape(self.pool4),
                                            num_classes=num_classes,
                                            debug=debug, name='upscore2',
                                            ksize=4, stride=2)
        # 将pool4层转换成同样类别数的输出
        self.score_pool4 = self._score_layer(self.pool4, "score_pool4",
                                             num_classes=num_classes)
        # 融合上采样一次的score_fr和pool4两层
        self.fuse_pool4 = tf.add(self.upscore2, self.score_pool4)

        # 上采样4倍
        # fuse_pool4层相当于原图的1/16
        # 对fuse_pool4层向上采样，增大了一倍
        self.upscore4 = self._upscore_layer(self.fuse_pool4,
                                            shape=tf.shape(self.pool3),
                                            num_classes=num_classes,
                                            debug=debug, name='upscore4',
                                            ksize=4, stride=2)
        # 将pool3层转换成同样类别数的输出
        self.score_pool3 = self._score_layer(self.pool3, "score_pool3",
                                             num_classes=num_classes)
        # 融合上采样一次的fuse_pool4和pool3两层
        self.fuse_pool3 = tf.add(self.upscore4, self.score_pool3)

        # 上采样32倍
        # fuse_pool3层相当于原图的1/8
        # 通过这一层相当于上采样回到了原图大小
        self.upscore32 = self._upscore_layer(self.fuse_pool3,
                                             shape=tf.shape(batch_images),
                                             num_classes=num_classes,
                                             debug=debug, name='upscore32',
                                             ksize=16, stride=8)

        self.pred_up = tf.argmax(self.upscore32, dimension=3)

        return tf.expand_dims(self.pred_up, dim=3), self.upscore32

    # 定义卷积层cd
    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            # 取得卷积核
            filt = self.get_conv_filter(name)
            # 定义卷积层
            conv = tf.nn.conv2d(input=bottom, filter=filt, strides=[1, 1, 1, 1], padding='SAME')
            # 取得偏置项
            bias = self.get_bias(name)
            # 求和
            bias_add = tf.nn.bias_add(conv, bias)
            # 激活函数
            relu = tf.nn.relu(bias_add)
            # 在summary中记录激活函数的输出
            _activation_summary(relu)
            return relu


    # 获取卷积核
    def get_conv_filter(self, name):
        # 用VGG的权重，初始化卷积核
        init = tf.constant_initializer(value=self.data_dict[name][0], dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        print('Layer name: %s' % name)
        print('Layer shape: %s' % str(shape))
        var = tf.get_variable(name="filter", initializer=init, shape=shape)
        # 把weights的二范数添加到正则化项，并且在summary中存入weights
        self._add_wd_and_summary(var, self.wd)
        return var


    # 获取偏置项
    def get_bias(self, name, num_classes=None):
        # 读取出偏置项的权重
        bias_weights = self.data_dict[name][1]
        shape = self.data_dict[name][1].shape

        # 如果是卷积层的最后一层，即给定了类别数量，要进一步根据输出类别数来reshape偏置项
        if num_classes is not None:
            bias_weights = self._num_classes_bias_reshape(bias_weights, shape[0], num_classes)
            shape = [num_classes]

        # 初始化偏置项变量
        init = tf.constant_initializer(value=bias_weights, dtype=tf.float32)
        var = tf.get_variable(name="biases", initializer=init, shape=shape)
        # 在summary中记录该变量
        _variable_summaries(var)
        return var


    # 定义最大池化层
    def _max_pool(self, bottom, name, debug=False):
        # 定义最大池化层
        pool = tf.nn.max_pool(value=bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

        if debug:
            pool = tf.Print(pool, [tf.shape(pool)],
                            message='Shape of %s' % name,
                            summarize=4, first_n=1)
        return pool

    # 依据VGG中的全连接层来定义全卷积层
    def _fc_layer(self, bottom, name, num_classes=None, relu=True, debug=False):
        with tf.variable_scope(name) as scope:
            # 获取卷积核
            # 需要判断是第几层全连接层，需要对原本VGG的参数做reshape
            if name == 'fc6':
                filt = self.get_fc_filter(name, [7, 7, 512, 4096])
            elif name == 'fc7':
                filt = self.get_fc_filter(name, [1, 1, 4096, 4096])
            elif name == 'score_fr':
                name = 'fc8'  # Name of score_fr layer in VGG Model
                filt = self.get_fc_filter(name, [1, 1, 4096, 1000], num_classes=num_classes)

            # 把filt的二范数添加到fc_wlosses，并且在summary中存入filt
            self._add_wd_and_summary(filt, self.wd, "fc_wlosses")

            # 定义卷积层
            conv = tf.nn.conv2d(input=bottom, filter=filt, strides=[1, 1, 1, 1], padding='SAME')
            bias = self.get_bias(name, num_classes=num_classes)
            bias_add = tf.nn.bias_add(conv, bias)
            # 是否使用激活函数
            if relu:
                relu = tf.nn.relu(bias_add)
            else:
                relu = bias_add
            # 在summary中记录激活函数的输出
            _activation_summary(relu)

            if debug:
                relu = tf.Print(relu, [tf.shape(relu)],
                                message='Shape of %s' % name,
                                summarize=4, first_n=1)
            return relu

    # 获取全卷积层的滤波器（由于VGG网络中的全连接层被修改成了全卷积层）
    def get_fc_filter(self, name, shape, num_classes=None):
        # 根据给定的滤波器shape，reshape权重
        print('Layer name: %s' % name)
        print('Layer shape: %s' % shape)
        weights = self.data_dict[name][0]
        weights = weights.reshape(shape)

        # 如果是最后一层，即给定了类别数量，需要进一步根据类别数量对权重进行reshape
        if num_classes is not None:
            weights = self._num_classes_filt_reshape(weights, shape, num_new=num_classes)

        # 初始化卷积核
        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        var = tf.get_variable(name="weights", initializer=init, shape=shape)

        return var

    # 根据类别数量对filt权重进行reshape
    # 这里的VGG网络输出的类别数量是1000，我们的task的类别数量往往是几十，所以这里的思想是把1000类的权重等分为几十份，然后求平均
    def _num_classes_filt_reshape(self, fweight, shape, num_new):
        num_orig = shape[3]
        shape[3] = num_new
        assert(num_new < num_orig)
        n_averaged_elements = num_orig//num_new
        avg_fweight = np.zeros(shape)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_fweight[:, :, :, avg_idx] = np.mean(
                fweight[:, :, :, start_idx:end_idx], axis=3)
        return avg_fweight

    # 根据类别数量对bias偏置项进行reshape
    # 这里的VGG网络输出的类别数量是1000，我们的task的类别数量往往是几十，所以这里的思想是把1000类的bias等分为几十份，然后求平均
    def _num_classes_bias_reshape(self, bweight, num_orig, num_new):
        n_averaged_elements = num_orig//num_new
        avg_bweight = np.zeros(num_new)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
        return avg_bweight

    # 定义一个卷积层,
    # 可以把某一层多通道的feature map转换成通道数为num_classes的score map
    # example:
    # 　　比如现在这一层的feature map一共有512张，但是task要求20类分类，因此这里定义了一个卷积操作可以把512通道输出为20通道
    def _score_layer(self, bottom, name, num_classes):
        with tf.variable_scope(name) as scope:
            # 获取输入通道数
            in_features = bottom.get_shape()[3].value
            shape = [1, 1, in_features, num_classes]
            # 设置产生不同层权重的正态分布对应的标准差
            if name == "score_fr":
                num_input = in_features
                stddev = (2 / num_input)**0.5
            elif name == "score_pool4":
                stddev = 0.001
            elif name == "score_pool3":
                stddev = 0.0001

            # 获取滤波器（由正态分布随机产生）
            w_decay = self.wd
            # 用截断的正态分布来初始化权重
            initializer = tf.truncated_normal_initializer(stddev=stddev)
            weights = tf.get_variable('weights', shape=shape, initializer=initializer)
            # 把weights的二范数添加到正则化项，并且在summary中存入weights
            self._add_wd_and_summary(weights, w_decay)
            # 定义卷积层
            conv = tf.nn.conv2d(input=bottom, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

            # 获取偏置项
            initializer = tf.constant_initializer(0.0)
            bias = tf.get_variable(name='biases', shape=[num_classes], initializer=initializer)
            # 在summary中记录该变量
            _variable_summaries(bias)
            # 求和
            bias_add = tf.nn.bias_add(conv, bias)

        # 在summary中记录该层的输出
        _activation_summary(bias_add)

        return bias_add

    # 定义一个上采样层，反卷积层或者说是转置卷积层
    def _upscore_layer(self, bottom, shape, num_classes, name, debug=False, ksize=4, stride=2):
        # 定义卷积步长
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            # 输入通道数
            in_features = bottom.get_shape()[3].value
            # 判断shape有没有定义，shape指的是希望上采样后得到的featuremap的大小
            if shape is None:
                # 如果shape没有定义，就自动计算一个
                in_shape = tf.shape(bottom)
                h = ((in_shape[1] - 1) * stride) + 1
                w = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], h, w, num_classes]
            else:
                # 注意：这里输出通道是num_classes
                new_shape = [shape[0], shape[1], shape[2], num_classes]
            output_shape = tf.stack(new_shape)

            # 滤波器的shape
            # logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
            f_shape = [ksize, ksize, num_classes, in_features]

            # 获取滤波器（双线性插值初始化）
            weights = self.get_deconv_filter(f_shape)
            self._add_wd_and_summary(weights, self.wd, "fc_wlosses")

            # 定义转置卷积层
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape, strides=strides, padding='SAME')

            if debug:
                deconv = tf.Print(deconv, [tf.shape(deconv)],
                                  message='Shape of %s' % name,
                                  summarize=4, first_n=1)

        _activation_summary(deconv)
        return deconv

    # 获取转置卷积滤波器的权重，双线性初始化
    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        height = f_shape[1]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        # 这里选择了不能训练
        var = tf.get_variable(name="up_filter", initializer=init, shape=weights.shape, trainable=False)

        return var

    # 添加Loss项，往summary中添加变量
    def _add_wd_and_summary(self, var, wd, collection_name=None):
        # 如果没有给定Loss添加到的位置，那么默认是添加到正则化项处
        if collection_name is None:
            collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection(collection_name, weight_decay)
        # 往summary中添加变量var
        _variable_summaries(var)
        return var


# 在summary中添加变量（通常是权重）
def _variable_summaries(var):
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        logging.info("Creating Summary for: %s" % name)
        with tf.name_scope('summaries'):
            # 添加均值
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name + '/mean', mean)
            # 添加标准差
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar(name + '/sttdev', stddev)
            # 添加最大最小值
            tf.summary.scalar(name + '/max', tf.reduce_max(var))
            tf.summary.scalar(name + '/min', tf.reduce_min(var))
            # 添加数据histogram
            tf.summary.histogram(name, var)


# 在summary中添加变量（通常是每一层的层输出）
def _activation_summary(x):
    tensor_name = x.op.name
    # 添加数据histogram
    tf.summary.histogram(tensor_name + '/activations', x)
    # 添加数据稀疏性
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))