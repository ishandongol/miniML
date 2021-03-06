import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from linearRegression.LinearRegression import LinearRegression


def linear_regression_runner():
    boston = load_boston()
    boston_data = pd.DataFrame(boston.data)
    boston_data.columns = boston.feature_names
    boston_data['PRICE'] = boston.target
    boston_data = (boston_data - boston_data.mean()) / boston_data.std()

    print(boston_data.head())

    Y = boston_data.iloc[:, 13:14].values
    X = boston_data.iloc[:, 0:13]
    ones = np.ones([len(X), 1])
    X = np.concatenate((ones, X), axis=1)

    X_train, X_rest, Y_train, Y_rest = train_test_split(X, Y, test_size=0.4)
    X_test, X_validate, Y_test, Y_validate = train_test_split(X_rest, Y_rest, test_size=0.5)

    print('Training Data: \t', len(X_train))
    print('Validation Data: \t', len(X_validate))
    print('Test Data: \t', len(X_test))

    alpha = 0.001  # Learning Rate
    lamda = 0.1  # regularization rate
    max_iterations = 10000
    W = np.zeros((1, len(X_train[0, :])))

    linearRegression = LinearRegression()
    Weight, Cost = linearRegression.train(X_train, Y_train, W, alpha, max_iterations, lamda)

    w = Weight.flatten()
    pandasW = pd.DataFrame(Weight.T)
    pandasW.columns = ["Theta"]
    print(pandasW)
    plt.scatter(Y_train, linearRegression.get_hypothesis())
    plt.xlabel("Prices: $Y_I$")
    plt.ylabel("Predicted Prices: $\hat{y}_i$")
    plt.title("Prices vs predicted prices: $Y_i$ vs $\hat{y}_i$ (Training)")
    plt.show()
    print('Cost')
    print('')
    print("Training Cost \t\t", Cost[len(Cost) - 1])
    print('')

    # cost_validation = linearRegression.validate(X_validate, Y_validate, Weight)
    # print("Validation Cost \t", cost_validation)
    # print('')
    # plt.scatter(Y_validate, linearRegression.get_hypothesis())
    # plt.xlabel("Prices: $Y_I$")
    # plt.ylabel("Predicted Prices: $\hat{y}_i$")
    # plt.title("Prices vs predicted prices: $Y_i$ vs $\hat{y}_i$ (Validation)")
    #
    # cost_test = linearRegression.test(X_test, Y_test, Weight)
    # print("Test Cost \t\t", cost_test)
    # plt.scatter(Y_test, linearRegression.get_hypothesis(), color='g')
    # plt.xlabel("Prices: $Y_I$")
    # plt.ylabel("Predicted Prices: $\hat{y}_i$")
    # plt.title("Prices vs predicted prices: $Y_i$ vs $\hat{y}_i$ (Test)")
    # plt.show()