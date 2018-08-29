from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras.backend as K
import keras
import glob
import imageio as magic
import numpy as np
from keras.optimizers import SGD


def run():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=1296))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(58, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    image_data = []
    labels = []
    for file_name in glob.iglob('/home/lognod/Desktop/nepali_characters/**/*.jpg', recursive=True):
        image_array = magic.imread(file_name, as_gray=True)
        label = int(file_name[-14:-11])
        labels.append(label)
        pixel_data = (255.0 - image_array.flatten()) / 255.0
        #     pixel_data = np.append(label,pixel_data)
        image_data.append(pixel_data)

    image_data = np.array(image_data)
    labels = np.array(labels)

    one_hot_labels = keras.utils.to_categorical(labels, 58)

    model.fit(image_data, one_hot_labels, epochs=1000, batch_size=128)

    model.save("ann3.h5")
    



