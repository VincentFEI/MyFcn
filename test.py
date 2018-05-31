import os
import datetime
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image
import model.fcn32_vgg as FCN32
import model.fcn16_vgg as FCN16
import model.fcn8_vgg as FCN8
from utils import ReadDataset
from utils import utlis

# 设置使用的GPU编号
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 参数列表
IMAGE_WIDTH = None
IMAGE_HEIGHT = None
BATCH_SIZE = None
NUM_CLASSES = None
MAX_ITERATION = None
LEARNING_RATE = None
KEEP_PROBABILITY = None
DEBUG = None
MODE = None
LOGS_DIR = None

# 命令行参数解析
flags = tf.flags
FLAGS = flags.FLAGS
# flags.DEFINE_integer("nums", 1, "input the quantities that you want to test")
flags.DEFINE_string("path", "/DATA/234/gxrao1/DeepLearning/dataset/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png", "input the path of test image")

def main(argv=None):

    # 从JSON文件中读取外部参数
    with open('params.json', 'r') as f:
        params_dict = json.load(f)

    global IMAGE_WIDTH
    IMAGE_WIDTH = params_dict["IMAGE_WIDTH"]
    global IMAGE_HEIGHT
    IMAGE_HEIGHT = params_dict["IMAGE_HEIGHT"]
    global BATCH_SIZE
    BATCH_SIZE = params_dict["BATCH_SIZE"]
    global NUM_CLASSES
    NUM_CLASSES = params_dict["NUM_CLASSES"]
    global MAX_ITERATION
    MAX_ITERATION = params_dict["MAX_ITERATION"]
    # MAX_ITERATION = int(1e5 + 1)
    global LEARNING_RATE
    LEARNING_RATE = params_dict["LEARNING_RATE"]
    global KEEP_PROBABILITY
    KEEP_PROBABILITY = params_dict["KEEP_PROBABILITY"]
    global DEBUG
    DEBUG = params_dict["DEBUG"]
    global MODE
    MODE = params_dict["MODE"]
    global LOGS_DIR
    LOGS_DIR = params_dict["LOGS_DIR"]


    # 构建计算图
    with tf.variable_scope('Graph') as scope:
        # 设置占位符
        images = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, 3], name="input_image")
        annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, 1], name="annotation")
        keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")

        # 构建网络模型
        # fcnvgg = FCN32.FCN32VGG()
        fcnvgg = FCN8.FCN8VGG()

        # 模型输出
        pred, logits = fcnvgg.build(images, num_classes=NUM_CLASSES, keepprob=keep_probability, debug=DEBUG)

        with tf.variable_scope('Loss'):
            # 计算交叉熵Loss
            entropy_loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                                          name="entropy")))
            # 添加交叉熵Loss到总Loss中
            tf.add_to_collection(tf.GraphKeys.LOSSES, entropy_loss)
            # 添加交叉熵Loss到summary
            tf.summary.scalar("entropy_loss", entropy_loss)

            # 正则化loss
            regularize_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            tf.summary.scalar("regularize_loss", regularize_loss)
            # fc_loss = tf.add_n(tf.get_collection("fc_wlosses"))
            # tf.summary.scalar("fc_loss", fc_loss)

            # 计算总Loss
            loss = entropy_loss + regularize_loss
            # 添加总Loss到summary
            tf.summary.scalar("whole_loss", loss)


    # 开启一个新会话
    with tf.Session() as sess:

        # 初始化Saver，用于存储模型参数
        saver = tf.train.Saver()
        # 初始化Summary_writer，用于记录summary
        summary_writer = tf.summary.FileWriter(LOGS_DIR, sess.graph)
        # 判断是否有训练过的模型，即ckpt文件，有的话直接读取
        ckpt = tf.train.get_checkpoint_state(LOGS_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")

        # 初始化所有参数
        sess.run(tf.global_variables_initializer())

        # 获取测试数据
        img_path = FLAGS.path
        test_images = utlis.read_single_image(img_path)
        test_images = np.reshape(test_images, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 3])

        # 启动测试过程
        feed_dict = {images: test_images, keep_probability: 1.0}
        predict = sess.run(pred, feed_dict=feed_dict)
        print(predict)
        print(predict.shape)
        imagemat = np.squeeze(predict)
        print(imagemat.shape)
        gray_img = PIL.Image.fromarray(np.uint8(imagemat))
        gray_img_path = img_path.split('.')[0] + "_test_gray.png"
        gray_img.save(gray_img_path)








if __name__ == "__main__":
    tf.app.run()


