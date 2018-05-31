import os
import datetime
import json
import numpy as np
import tensorflow as tf
import PIL.Image
import model.fcn32_vgg as FCN32
import model.fcn16_vgg as FCN16
import model.fcn8_vgg as FCN8

from utils import ReadDataset
from utils import utlis
from utils import ReadPascalVOC
from utils import BatchDatsetReader
from utils import ReadMITSceneParing

# 设置使用的GPU编号
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 参数列表
IMAGE_WIDTH = None
IMAGE_HEIGHT = None
IMAGE_SIZE = None
BATCH_SIZE = None
NUM_CLASSES = None
MAX_ITERATION = None
LEARNING_RATE = None
KEEP_PROBABILITY = None
DEBUG = None
MODE = None
LOGS_DIR = None
DATASET = None
PASCAL_DIR = None
MIT_DIR = None

def main(argv=None):
    # 从JSON文件中读取外部参数
    with open('params.json', 'r') as f:
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
    global DEBUG
    DEBUG = params_dict["DEBUG"]
    global MODE
    MODE = params_dict["MODE"]
    global LOGS_DIR
    LOGS_DIR = params_dict["LOGS_DIR"]
    global DATASET
    DATASET = params_dict["DATASET"]
    global PASCAL_DIR
    PASCAL_DIR = params_dict["PASCAL_DIR"]
    global MIT_DIR
    MIT_DIR = params_dict["MIT_DIR"]

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
        image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
        if MODE == 'train':
            train_dataset = BatchDatsetReader.BatchDatset(train_records, image_options)
        val_dataset = BatchDatsetReader.BatchDatset(valid_records, image_options)
    elif DATASET == "PASCAL":
        # Pascal VOC Dataset
        train_records, valid_records = ReadPascalVOC.read_dataset(PASCAL_DIR)
        image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
        if MODE == 'train':
            train_dataset = BatchDatsetReader.BatchDatset(train_records, image_options)
        val_dataset = BatchDatsetReader.BatchDatset(valid_records, image_options)
    else:
        print("Error : Void 'DATASET', 'DATASET' in json file should be 'MIT' or 'PASCAL'.")
        return


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

    # 开启一个新会话
    with tf.Session() as sess:

        # 初始化Saver，用于存储模型参数，最多记录11个模型
        saver = tf.train.Saver(max_to_keep=11)
        # 初始化Summary_writer，用于记录summary
        summary_writer = tf.summary.FileWriter(LOGS_DIR, sess.graph)
        # 初始化所有参数
        sess.run(tf.global_variables_initializer())
        # 判断是否有训练过的模型，即ckpt文件，有的话直接读取
        ckpt = tf.train.get_checkpoint_state(LOGS_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")

        # 判断是训练模式还是测试模式
        if MODE == "train":
            # 训练迭代循环
            for itr in range(MAX_ITERATION):
                # 获取训练数据
                train_images, train_annotations = train_dataset.get_next_batch(BATCH_SIZE)
                train_images = np.reshape(train_images, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 3])
                train_annotations = np.reshape(train_annotations, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
                # 启动训练过程
                feed_dict = {images: train_images, annotation: train_annotations, keep_probability: KEEP_PROBABILITY}
                sess.run(train_op, feed_dict=feed_dict)
                # 10次训练后，计算一次loss，记录一次summary
                if itr % 10 == 0:
                    train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                    print("Step: %d, Train_loss:%g" % (itr, train_loss))
                    summary_writer.add_summary(summary_str, itr)
                # 500次训练后，计算一次验证集上的loss
                if itr % 500 == 0:
                    valid_images, valid_annotations = val_dataset.get_next_batch(BATCH_SIZE)
                    # valid_images = np.reshape(valid_images, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 3])
                    # valid_annotations = np.reshape(valid_annotations, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
                    feed_dict = {images: valid_images, annotation: valid_annotations, keep_probability: 1.0}
                    valid_loss, valid_preds = sess.run([loss, pred], feed_dict=feed_dict)
                    print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                    valid_annotations = np.squeeze(valid_annotations, axis=3)
                    valid_preds = np.squeeze(valid_preds, axis=3)

                    for idx in range(BATCH_SIZE):
                        print("Saved image: %d" % idx)
                        predimg = valid_preds[idx,:,:]
                        predimagemat = np.squeeze(predimg)
                        predimage = PIL.Image.fromarray(np.uint8(predimagemat))

                        annoimg = valid_annotations[idx, :, :]
                        annoimagemat = np.squeeze(annoimg)
                        annoimage = PIL.Image.fromarray(np.uint8(annoimagemat))

                        if DATASET == "MIT":
                            predimgpath = "MIT_pred_" + str(idx) + ".jpg"
                            annoimgpath = "MIT_anno_" + str(idx) + ".jpg"
                        elif DATASET == "PASCAL":
                            predimgpath = "PASCAL_pred_" + str(idx) + ".jpg"
                            annoimgpath = "PASCAL_anno_" + str(idx) + ".jpg"

                        predimage.save(predimgpath)
                        annoimage.save(annoimgpath)


                # 5000次训练后，记录模型参数
                if itr % 5000 == 0:
                    saver.save(sess, LOGS_DIR + "model.ckpt", itr)

        elif MODE == "test":

            # 获取测试数据
            # # CityScapes Dataset
            # test_image_path = "/DATA/234/gxrao1/DeepLearning/dataset/leftImg8bit/test"
            # test_dataset = ReadDataset.ReadDataset(image_path=test_image_path)
            # test_images = test_dataset.get_next_batch(BATCH_SIZE)
            test_images = val_dataset.get_next_batch(BATCH_SIZE)
            # 启动测试过程
            feed_dict = {images: test_images, keep_probability: 1.0}
            predict = sess.run(pred, feed_dict=feed_dict)
            print(predict)
            print(predict.shape)
            imagemat = np.squeeze(predict)
            print(imagemat.shape)
            image = PIL.Image.fromarray(np.uint8(imagemat))
            image.save("test.jpg")


if __name__ == "__main__":
    tf.app.run()


