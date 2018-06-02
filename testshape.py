from utils import BatchDatsetReader
from utils import ReadMITSceneParing
train_records, valid_records = ReadMITSceneParing.read_dataset("/DATA/234/gxrao1/DeepLearning/FCN.tensorflow/Data_zoo/MIT_SceneParsing")
# image_options = {'resize': True, 'resize_size': 224}
image_options = {'resize': False}
train_dataset = BatchDatsetReader.BatchDatset(train_records[:10], image_options)
train_images, train_annotations = train_dataset.get_next_batch(2)
print(train_images.shape)
print(train_annotations.shape)
print(train_images)
print(train_annotations)