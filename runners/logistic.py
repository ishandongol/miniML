import numpy as np
import glob
import imageio as magic
import pandas as pd
from sklearn.model_selection import train_test_split

from logisticRegression.LogisticRegression import LogisticRegression


def logistic_regression_runner():
    image_data = []
    label_list = []
    for file_name in glob.iglob('/home/lognod/Desktop/nhcd/numerals/**/*.jpg', recursive=True):
        image_array = magic.imread(file_name, as_gray=True)
        label_list.append(file_name[-12:-11])
        label = int(file_name[-12:-11])
        pixel_data = (255.0 - image_array.flatten()) / 256.0
        pixel_data = np.append(label, pixel_data)
        image_data.append(pixel_data)

    image_data = np.array(image_data)
    np.random.shuffle(image_data)
    image_data_pd = pd.DataFrame(image_data)
    image_data_pd.head()

    X = image_data_pd.iloc[:, 1:]
    ones = np.ones([len(X), 1])
    X = np.concatenate((ones, X), axis=1)
    Y = image_data_pd.iloc[:, 0:1].values
    print(X.shape)
    print(Y)
    X_train, X_rest, Y_train, Y_rest = train_test_split(X, Y, test_size=0.4)
    X_validate, X_test, Y_validate, Y_test = train_test_split(X_rest, Y_rest, test_size=0.5)

    logistic_regression = LogisticRegression()
    weight_list = []
    cost_list = []
    label_list = set(label_list)

    for i in range(len(label_list)):
        W = np.zeros((1, len(X_train[0, :])))
        print("Learning: ", float(i))
        Y_train_one = (Y_train == float(i)).astype(int)
        weight, cost = logistic_regression.train(X_train, Y_train_one, W, 0.01, 100, 0)
        weight_list.append(weight.flatten())
        cost_list.append(cost)

    print(weight_list)

    weights = np.array(weight_list)
    weights_data = pd.DataFrame(weights)
    weights_data.to_csv("/home/lognod/mini_logistic3.csv")

    pd.DataFrame(cost_list).to_csv("/home/lognod/cost_logistic_3.csv")

    logistic_regression = LogisticRegression()
    weights = pd.read_csv("/home/lognod/mini_logistic3.csv")
    y = []
    for i in range(10):
        y.append(logistic_regression.predict(X_validate[20, :], weights.iloc[i, 1:]))

    prediction = np.where(y == np.amax(y))
    print("It is: ",prediction[0])

    test = X_validate[20, 1:]

    image = test.reshape(-1, 28)
    magic.imwrite('/home/lognod/predicted.png', image)
