import numpy as np
from PIL import Image
import os
import random

# 读取数据集的程序
class ReadDataset:
    # 指示当前读取到了图片名称列表的具体位置
    batch_offset = 0
    # 指示当前是第几个epochs
    epochs_completed = 0

    # 初始化
    def __init__(self, image_path = None, label_path = None):

        if image_path:
            self.image_filelists = self._import_dataset_lists(image_path)
        else:
            self.image_filelists = None

        if label_path:
            self.label_filelists = self._import_dataset_lists(label_path, type="labelTrainIdsModify")
        else:
            self.label_filelists = None


    # 图片名称读取到列表中来
    def _import_dataset_lists(self, path, type=None):

        filelists = []
        parentpath = path
        filedirs = os.listdir(parentpath)
        for filedir in filedirs:
            # print(filedir)
            childpath = parentpath + "/" + filedir
            files = os.listdir(childpath)
            for file in files:
                # print(file)
                if not type or type in file:
                    filename = childpath + "/" + file
                    filelists.append(filename)
        sorted_filelists = sorted(filelists, key=str.lower)
        return sorted_filelists

    # 获取下一个用于训练的Batch
    def get_next_batch(self, batch_size):
        # 拿到下一个batch的lists
        start = self.batch_offset
        self.batch_offset += batch_size

        # 如果batch_offset大于了样本总数，开始全新的一轮
        if self.batch_offset > len(self.image_filelists):
            # 结束旧一轮
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # 打乱顺序
            if self.label_filelists:
                combine_lists = list(zip(self.image_filelists, self.label_filelists))
                random.shuffle(combine_lists)
                self.image_filelists[:], self.label_filelists[:] = zip(*combine_lists)
            else:
                random.shuffle(self.image_filelists)

            # 开始新一轮
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset


        batch_image_lists = self.image_filelists[start:end]
        images = self._import_images(batch_image_lists)

        if self.label_filelists:
            batch_label_lists = self.label_filelists[start:end]
            labels = self._import_images(batch_label_lists)
            return images, labels
        else:
            print(batch_image_lists)
            return images


    # 输入图片名称的列表，输出图片组成的矩阵
    def _import_images(self, lists):
        images = np.array([np.array(Image.open(file)) for file in lists])
        return images



    # def get_random_batch(self, batch_size):
    #     indexes = np.random.randint(0, self.train_image_filelists.shape[0], size=[batch_size]).tolist()
    #     return self.train_image_filelists[indexes], self.train_label_filelists[indexes]