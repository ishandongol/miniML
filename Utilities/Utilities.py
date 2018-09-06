import glob
import cv2 as cv
import numpy as np
import argparse

def load_image(url,type):
    image_data = []
    labels = []
    for file_name in glob.iglob(url, recursive=True):
        image_array = cv.imread(file_name, 0)
        label = int(file_name[-14:-11])
        labels.append(label)
        image_data.append(image_array)


    image_data = np.array(image_data)

    if (str2bool(type)):
        print("inverting & normalizing...")
        image_data = (255.0 - image_data) / 255.0
    else:
        print("normalizing...")
        image_data = image_data/255.0

    labels = np.array(labels)

    return image_data,labels

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

