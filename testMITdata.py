
import numpy as np
from utils import BatchDatsetReader
from utils import ReadMITSceneParing

train_records, valid_records = ReadMITSceneParing.read_dataset("/home/vincentfei/shell/MIT/")
image_options = {'resize': False}
train_dataset = BatchDatsetReader.BatchDatset(train_records, image_options)
for i in range(30):
    train_images, train_annotations = train_dataset.get_next_batch(1)
    train_images_list = list(train_images)
    train_annotations_lsit = list(train_annotations)
    train_images = np.asarray(train_images_list)
    train_annotations = np.asarray(train_annotations_lsit)
    print("Iter : %d" % i )
    print(train_images.shape)
    print(train_annotations.shape)
