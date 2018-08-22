import numpy as np

from logisticRegression.LogisticRegression import LogisticRegression


def logistic_regression_runner():
    logistic_regression = LogisticRegression()
    cost = logistic_regression.validate(np.array([2, 2]), np.array([3, 2]), np.array([[2, 1]]))
    print("Cost: ", cost.flatten())