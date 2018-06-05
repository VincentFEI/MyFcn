import os
import random
from six.moves import cPickle as pickle


def read_dataset(data_dir):
    pickle_filename = "PASCALVOC.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        result = create_image_lists(data_dir)
        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found PASCAL pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        del result

    return training_records, validation_records



def create_image_lists(data_dir):
    train_data_dir = os.path.join(data_dir, "ImageSets/Segmentation/train.txt")
    valid_data_dir = os.path.join(data_dir, "ImageSets/Segmentation/val.txt")

    train_data_file = open(train_data_dir)
    train_data_list = train_data_file.readlines()
    train_records = []
    for file in train_data_list:
        filename = file.split("\n")[0]
        image = os.path.join(data_dir, "JPEGImages/", filename + ".jpg")
        annotation = os.path.join(data_dir, "SegmentationClassPreCode1/", filename + ".png")
        record = {'image': image, 'annotation': annotation, 'filename': filename}
        train_records.append(record)

    valid_data_file = open(valid_data_dir)
    valid_data_list = valid_data_file.readlines()
    random.shuffle(valid_data_list)
    train_data_list_new = valid_data_list[:949]
    valid_data_list_new = valid_data_list[949:]

    for file in train_data_list_new:
        filename = file.split("\n")[0]
        image = os.path.join(data_dir, "JPEGImages/", filename + ".jpg")
        annotation = os.path.join(data_dir, "SegmentationClassPreCode1/", filename + ".png")
        record = {'image': image, 'annotation': annotation, 'filename': filename}
        train_records.append(record)

    valid_records = []
    for file in valid_data_list_new:
        filename = file.split("\n")[0]
        image = os.path.join(data_dir, "JPEGImages/", filename + ".jpg")
        annotation = os.path.join(data_dir, "SegmentationClassPreCode1/", filename + ".png")
        record = {'image': image, 'annotation': annotation, 'filename': filename}
        valid_records.append(record)

    image_list = {'training':train_records, 'validation':valid_records}
    return image_list