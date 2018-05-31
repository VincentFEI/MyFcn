import os
import random
from utils import BatchDatsetReader

def read_dataset(data_dir):
    train_data_dir = os.path.join(data_dir, "ImageSets/Segmentation/train.txt")
    valid_data_dir = os.path.join(data_dir, "ImageSets/Segmentation/val.txt")

    train_data_file= open(train_data_dir)
    train_data_list = train_data_file.readlines()
    train_records = []
    for file in train_data_list:
        filename = file.split("\n")[0]
        image = os.path.join(data_dir,"JPEGImages/",filename+".jpg")
        annotation = os.path.join(data_dir,"SegmentationClassPreCode1/",filename+".png")
        record = {'image': image, 'annotation': annotation, 'filename': filename}
        train_records.append(record)

    valid_data_file= open(valid_data_dir)
    valid_data_list = valid_data_file.readlines()
    random.shuffle(valid_data_list)
    train_data_list_new = valid_data_list[:949]
    valid_data_list_new = valid_data_list[949:]

    for file in train_data_list_new:
        filename = file.split("\n")[0]
        image = os.path.join(data_dir,"JPEGImages/",filename+".jpg")
        annotation = os.path.join(data_dir,"SegmentationClassPreCode1/",filename+".png")
        record = {'image': image, 'annotation': annotation, 'filename': filename}
        train_records.append(record)

    valid_records = []
    for file in valid_data_list_new:
        filename = file.split("\n")[0]
        image = os.path.join(data_dir,"JPEGImages/",filename+".jpg")
        annotation = os.path.join(data_dir,"SegmentationClassPreCode1/",filename+".png")
        record = {'image': image, 'annotation': annotation, 'filename': filename}
        valid_records.append(record)

    return train_records, valid_records


