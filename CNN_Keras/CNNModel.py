import CNN_Keras.Parameters as Params
from keras.models import Sequential
from keras.layers import Dense,Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
from Utilities.Utilities import load_image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from Callbacks.CustomCallback import TrainValTensorBoard
import time
from keras import backend as K


class CNNModel:

    __input_shape = None

    def __init__(self):
        print("Convolutional Neural Network")


    def create_model(self):
        model = Sequential()
        model.add(Conv2D(Params.ConvLayers["filter"],Params.ConvLayers["kernel_size"],padding=Params.ConvLayers["padding"],activation=Params.ConvLayers["activation"],input_shape=self.__input_shape))
        model.add(Conv2D(Params.ConvLayers["filter"],Params.ConvLayers["kernel_size"],activation=Params.ConvLayers["activation"]))
        model.add(MaxPooling2D(pool_size=Params.ConvLayers["pool_size"]))
        model.add(Dropout(Params.ConvLayers["dropout"]))

        model.add(
            Conv2D(Params.ConvLayers["filter"]*2, Params.ConvLayers["kernel_size"], padding=Params.ConvLayers["padding"],
                   activation=Params.ConvLayers["activation"]))
        model.add(Conv2D(Params.ConvLayers["filter"]*2, Params.ConvLayers["kernel_size"],
                         activation=Params.ConvLayers["activation"]))
        model.add(MaxPooling2D(pool_size=Params.ConvLayers["pool_size"]))
        model.add(Dropout(Params.ConvLayers["dropout"]))

        model.add(
            Conv2D(Params.ConvLayers["filter"] * 2, Params.ConvLayers["kernel_size"],
                   padding=Params.ConvLayers["padding"],
                   activation=Params.ConvLayers["activation"]))
        model.add(Conv2D(Params.ConvLayers["filter"] * 2, Params.ConvLayers["kernel_size"],
                         activation=Params.ConvLayers["activation"]))
        model.add(MaxPooling2D(pool_size=Params.ConvLayers["pool_size"]))
        model.add(Dropout(Params.ConvLayers["dropout"]))

        model.add(Flatten())
        model.add(Dense(Params.CoreLayers["nodes"],activation=Params.CoreLayers["activation"]))
        model.add(Dropout(Params.CoreLayers["dropout"]))
        model.add(Dense(Params.CoreLayers["output_node"],activation=Params.CoreLayers["activation_output"]))
        print(model.summary())

        return model

    def run(self,url,output,type):
        images, labels = load_image(url,type)

        x_train, x_validate, y_train, y_validate = train_test_split(images, labels,
                                                                    test_size=0.2, shuffle=True,
                                                                    stratify=labels)
        img_rows = img_cols = Params.Parameters["image_dim"]

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_validate = x_validate.reshape(x_validate.shape[0], 1, img_rows, img_cols)
            self.__input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_validate = x_validate.reshape(x_validate.shape[0], img_rows, img_cols, 1)
            self.__input_shape = (img_rows, img_cols, 1)

        model = self.create_model()
        adam = Adam(lr=Params.Parameters["lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer=adam, loss=Params.Parameters["loss"],metrics=['accuracy'])

        one_hot_labels = to_categorical(y_train, Params.CoreLayers["output_node"])
        one_hot_validate = to_categorical(y_validate, Params.CoreLayers["output_node"])

        tbCallBack = TrainValTensorBoard(
            log_dir='./Graph/nepali-{0}-{1}-{2}-'.format(Params.Parameters, Params.CoreLayers, int(time.time())),
            histogram_freq=0, write_images=True)
        model.fit(x_train, one_hot_labels, epochs=Params.Parameters["epoch"], batch_size=Params.Parameters["batch_size"],
                  validation_data=(x_validate, one_hot_validate), callbacks=[tbCallBack])

        model.save(output+"cnn-{}.h5".format(time.time()))

