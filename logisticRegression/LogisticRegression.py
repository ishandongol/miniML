import numpy as np


class LogisticRegression:

    def __init__(self):
        print("Logistic Regression")

    def get_sigmoid(self, X, W):
        return 1 / (1 + np.exp(- self.get_hypothesis(X, W)))

    def get_hypothesis(self, X, W):
        return X @ W.T

    def __get_cost(self, X, Y, W, lamda):
        return -(1.0 / len(X)) * np.sum(
            (Y * np.log(self.get_sigmoid(X, W))) + ((1 - Y) * np.log(1 - self.get_sigmoid(X, W))))

    def __get_gradient(self, X, Y, W, lamda):
        return (1.0 / len(X)) * (np.sum(X * (self.get_hypothesis(X, W) - Y), axis=0) + (lamda * W))

    def __logistic_regression(self, X, Y, W, alpha, max_iterations, lamda):

        for i in range(max_iterations):

            W = W - alpha * self.__get_gradient(X, Y, W, lamda)
            cost = self.__get_cost(X, Y, W, lamda)

            if i % 100 == 0:
                print("Cost: ", cost)

        return W, cost

    def train(self, X, Y, W, alpha, max_iterations, lamda=0):
        return self.__logistic_regression(X, Y, W, alpha, max_iterations, lamda)

    def validate(self, X, Y, W):
        return self.__get_cost(X, Y, W, 0)

    def test(self, X, Y, W, lamda=0):
        return self.__get_cost(X, Y, W, 0)