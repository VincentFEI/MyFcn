import sys
import os
import datetime
import json
import numpy as np
import tensorflow as tf
import PIL.Image
import scipy.misc as misc

import model.fcn32_vgg as FCN32
import model.fcn16_vgg as FCN16
import model.fcn8_vgg as FCN8
# import model.fcn8_vgg_notrain as FCN8

from utils import ReadDataset
from utils import utlis
from utils import ReadPascalVOC
from utils import BatchDatsetReader
from utils import ReadMITSceneParing

# 设置使用的GPU编号
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# 参数列表
IMAGE_WIDTH = None
IMAGE_HEIGHT = None
IMAGE_SIZE = None
BATCH_SIZE = None
NUM_CLASSES = None
MAX_ITERATION = None
LEARNING_RATE = None
KEEP_PROBABILITY = None
NO_RESIZE = None
DEBUG = None
MODE = None
LOGS_DIR = None
RESULT_DIR = None
DATASET = None
PASCAL_DIR = None
MIT_DIR = None

def main(argv=None):

    ## *****************************************************************************************************************
    ## *****************************************************************************************************************
    ## *****************************************************************************************************************
    #  Part one : 读取参数文件，数据文件
    ## *****************************************************************************************************************
    if len(sys.argv) > 1:
        paramsfile = sys.argv[1]
    else:
        paramsfile = 'params.json'

    # 从JSON文件中读取外部参数
    with open(paramsfile, 'r') as f:
        params_dict = json.load(f)

    global IMAGE_WIDTH
    IMAGE_WIDTH = params_dict["IMAGE_WIDTH"]
    global IMAGE_HEIGHT
    IMAGE_HEIGHT = params_dict["IMAGE_HEIGHT"]
    global IMAGE_SIZE
    IMAGE_SIZE = params_dict["IMAGE_SIZE"]
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
    global NO_RESIZE
    NO_RESIZE = params_dict["NO_RESIZE"]
    global DEBUG
    DEBUG = params_dict["DEBUG"]
    global MODE
    MODE = params_dict["MODE"]
    global LOGS_DIR
    LOGS_DIR = params_dict["LOGS_DIR"]
    global RESULT_DIR
    RESULT_DIR = params_dict["RESULT_DIR"]
    global DATASET
    DATASET = params_dict["DATASET"]
    global PASCAL_DIR
    PASCAL_DIR = params_dict["PASCAL_DIR"]
    global MIT_DIR
    MIT_DIR = params_dict["MIT_DIR"]


    # 读取数据

    # # CityScapes Dataset
    # train_image_path = "/DATA/234/gxrao1/DeepLearning/dataset/leftImg8bit/train"
    # train_label_path = "/DATA/234/gxrao1/DeepLearning/dataset/gtFine/train"
    # train_dataset = ReadDataset.ReadDataset(image_path=train_image_path, label_path=train_label_path)
    #
    # val_image_path = "/DATA/234/gxrao1/DeepLearning/dataset/leftImg8bit/val"
    # val_label_path = "/DATA/234/gxrao1/DeepLearning/dataset/gtFine/val"
    # val_dataset = ReadDataset.ReadDataset(image_path=val_image_path, label_path=val_label_path)

    if DATASET == "MIT":
        # MIT SceneParsing Dataset
        train_records, valid_records = ReadMITSceneParing.read_dataset(MIT_DIR)

        if NO_RESIZE == True:
            image_options = {'resize': False}
        else:
            image_options = {'resize': True, 'resize_size': IMAGE_SIZE}


        val_dataset = BatchDatsetReader.BatchDatset(valid_records, image_options)

    elif DATASET == "PASCAL":
        # Pascal VOC Dataset
        train_records, valid_records = ReadPascalVOC.read_dataset(PASCAL_DIR)

        if NO_RESIZE == True:
            image_options = {'resize': False}
        else:
            image_options = {'resize': True, 'resize_size': IMAGE_SIZE}

        val_dataset = BatchDatsetReader.BatchDatset(valid_records, image_options)

    else:
        print("Error : Void 'DATASET', 'DATASET' in json file should be 'MIT' or 'PASCAL'.")
        return

    ## *****************************************************************************************************************
    ## *****************************************************************************************************************
    ## *****************************************************************************************************************
    #  Part two : 构建计算图
    ## *****************************************************************************************************************
    with tf.variable_scope('Graph') as scope:
        # 设置占位符
        if NO_RESIZE == True:
            images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")
            annotation = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="annotation")
        else:
            images = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, 3], name="input_image")
            annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, 1], name="annotation")


        keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")

        # 构建网络模型
        fcnvgg = FCN8.FCN8VGG()

        # 模型输出
        pred, logits = fcnvgg.build(images, num_classes=NUM_CLASSES, keepprob=keep_probability, debug=DEBUG)

        # 添加图片到summary
        tf.summary.image("input_image", images, max_outputs=2)
        tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
        tf.summary.image("pred_annotation", tf.cast(pred, tf.uint8), max_outputs=2)

        with tf.variable_scope('Loss'):
            # 计算交叉熵Loss
            if DATASET == "PASCAL":
                entropy_loss = tf.reduce_mean((utlis.sparse_softmax_cross_entropy_ignore_labels(logits=logits,
                                                                                              labels=tf.squeeze(
                                                                                                  annotation,
                                                                                                  squeeze_dims=[3]),
                                                                                              name="entropy")))
            else:
                entropy_loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                                              labels=tf.squeeze(
                                                                                                  annotation,
                                                                                                  squeeze_dims=[3]),
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

        # 定义训练优化操作
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        grads = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads)

        # 定义记录操作
        summary_op = tf.summary.merge_all()

    ## *****************************************************************************************************************
    ## *****************************************************************************************************************
    ## *****************************************************************************************************************
    #  Part three : 开启一个新会话
    ## *****************************************************************************************************************
    with tf.Session() as sess:

        # 初始化Saver，用于存储模型参数，最多记录11个模型
        saver = tf.train.Saver(max_to_keep=11)
        # 初始化Summary_writer，用于记录summary
        summary_writer = tf.summary.FileWriter(LOGS_DIR, sess.graph)
        # 初始化所有参数
        sess.run(tf.global_variables_initializer())

        model_index_list = [0,2,4,6,8]
        result_dir_list = ["pascalresult_50000", "pascalresult_55000", "pascalresult_60000", "pascalresult_65000",
                           "pascalresult_70000", "pascalresult_75000", "pascalresult_80000", "pascalresult_85000",
                           "pascalresult_90000", "pascalresult_95000"]
        # for model_index in model_index_list:

        # 判断是否有训练过的模型，即ckpt文件，有的话直接读取
        ckpt = tf.train.get_checkpoint_state(LOGS_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            model_index = 2
            saver.restore(sess, ckpt.all_model_checkpoint_paths[model_index])
            print("Model path: " + LOGS_DIR + ckpt.all_model_checkpoint_paths[model_index])
            print("Model restored...")
        ## *************************************************************************************************************
        RESULT_DIR = result_dir_list[model_index]

        if DATASET == "MIT":
            for test_idx in range(2000):
                print("MIT Test image: %d" % test_idx)
                test_images, test_annotations = val_dataset.get_next_batch(1)

                if NO_RESIZE == True:
                    # 这一步只在数据不resize的时候使用
                    test_images_list = list(test_images)
                    test_annotations_lsit = list(test_annotations)
                    test_images = np.asarray(test_images_list)
                    test_annotations = np.asarray(test_annotations_lsit)

                feed_dict = {images: test_images, annotation: test_annotations, keep_probability: 1.0}
                test_preds = sess.run(pred, feed_dict=feed_dict)

                test_images = np.squeeze(test_images)
                test_annotations = np.squeeze(test_annotations)
                test_preds = np.squeeze(test_preds)

                test_images_path = RESULT_DIR + "MIT_img_"  + str(test_idx) + ".png"
                test_annos_path  = RESULT_DIR + "MIT_anno_" + str(test_idx) + ".png"
                test_preds_path  = RESULT_DIR + "MIT_pred_" + str(test_idx) + ".png"

                misc.imsave(test_images_path, test_images.astype(np.uint8))
                misc.imsave(test_annos_path, test_annotations.astype(np.uint8))
                misc.imsave(test_preds_path, test_preds.astype(np.uint8))

        elif DATASET == "PASCAL":
            for test_idx in range(500):
                print("PASCAL Test image: %d" % test_idx)
                test_images, test_annotations = val_dataset.get_next_batch(1)

                if NO_RESIZE == True:
                    # 这一步只在数据不resize的时候使用
                    test_images_list = list(test_images)
                    test_annotations_lsit = list(test_annotations)
                    test_images = np.asarray(test_images_list)
                    test_annotations = np.asarray(test_annotations_lsit)

                feed_dict = {images: test_images, annotation: test_annotations, keep_probability: 1.0}
                test_preds = sess.run(pred, feed_dict=feed_dict)

                test_images = np.squeeze(test_images)
                test_annotations = np.squeeze(test_annotations)
                test_preds = np.squeeze(test_preds)

                test_images_path = RESULT_DIR + "PASCAL_img_"  + str(test_idx) + ".png"
                test_annos_path  = RESULT_DIR + "PASCAL_anno_" + str(test_idx) + ".png"
                test_preds_path  = RESULT_DIR + "PASCAL_pred_" + str(test_idx) + ".png"

                misc.imsave(test_images_path, test_images.astype(np.uint8))
                misc.imsave(test_annos_path, test_annotations.astype(np.uint8))
                misc.imsave(test_preds_path, test_preds.astype(np.uint8))


if __name__ == "__main__":
    tf.app.run()


