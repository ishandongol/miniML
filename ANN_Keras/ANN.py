from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras
import glob
import cv2 as cv
import numpy as np
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import time
from Callbacks.CustomCallback import TrainValTensorBoard

from ANN_Keras.Parameters import Parameters, LayerParameters

def run():

    model = Sequential()
    model.add(Dense(LayerParameters["nodes"], activation='relu', input_dim=LayerParameters["input_dim"]))
    model.add(Dropout(LayerParameters["dropout"]))
    for i in range(LayerParameters["hidden_layers"]):
        model.add(Dense(LayerParameters["nodes"], activation='relu'))
        model.add(Dropout(LayerParameters["dropout"]))
        print(i)
    model.add(Dense(LayerParameters["output_node"], activation='softmax'))
    sgd = SGD(lr=Parameters["lr"], decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    image_data = []
    labels = []
    count = 0
    for file_name in glob.iglob('/home/lognod/nepali_characters/**/*.jpg', recursive=True):
        image_array = cv.imread(file_name, 0)
        label = int(file_name[-14:-11])
        pixel_data = (255-image_array) / 255.0
        pixel_data = np.append(label,pixel_data.flatten())
        image_data.append(pixel_data)
        count +=1

    image_data = np.array(image_data)
    # np.random.shuffle(image_data)

    x_train, x_validate, y_train, y_validate = train_test_split(image_data[:, 1:], image_data[:, 0:1], test_size=0.2,shuffle=True,stratify=image_data[:,0:1])

    one_hot_labels = keras.utils.to_categorical(y_train, 58)
    one_hot_validate = keras.utils.to_categorical(y_validate, 58)

    tbCallBack = TrainValTensorBoard(log_dir='./Graph/nepali-{0}-{1}-{2}-'.format(Parameters,LayerParameters,int(time.time())), histogram_freq=0, write_images=True)
    model.fit(x_train, one_hot_labels, epochs=Parameters["epoch"], batch_size=Parameters["batch_size"],validation_data=(x_validate,one_hot_validate), callbacks=[tbCallBack])
    model.save("/home/lognod/PycharmProjects/nepali-handwriting-recognitin/ann_final.h5")




