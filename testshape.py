import numpy as np
from utils import BatchDatsetReader
from utils import ReadPascalVOC
train_records, valid_records = ReadPascalVOC.read_dataset("/home/vincentfei/PycharmProjects/PASCALVOC/VOCdevkit/VOC2012/")
# image_options = {'resize': True, 'resize_size': 224}
image_options = {'resize': False}
train_dataset = BatchDatsetReader.BatchDatset(train_records[:10], image_options)
train_images, train_annotations = train_dataset.get_next_batch(1)
print(train_images.shape)
print(train_annotations.shape)

print(type(train_images))
print(type(train_images[0]))
print(train_images[0].shape)
a = list(train_images)
aa = np.asarray(a)
print(aa.shape)




