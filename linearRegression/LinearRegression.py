import numpy as np

class LinearRegression:

    __hypothesis = None

    def __init__(self):
        print("MiniML")

    def linearEquation(self,X,W):
        self.__hypothesis = X @ W.T
        return self.__hypothesis

    def calculateCost(self,X,Y,W,lamda=0):
        inner = np.power(((self.linearEquation(X,W))-Y),2) + (lamda * np.sum(np.power(W,2)))
        return (np.sum(inner)/(2*len(X)))


    def linearRegression(self,X,Y,W,alpha,lamda,max_iterations):
        cost = np.zeros(max_iterations)

        for i in range(max_iterations):
            W = W - (alpha/len(X)) * (np.sum(X * (X @ W.T - Y), axis=0)+(lamda * W))
            cost[i] = self.calculateCost(X,Y,W,lamda)
            if(i % 1000 == 0):
                print("Cost", cost[i])

        return W,cost

    def getHypothesis(self):
        return self.__hypothesis

    def train(self,X,Y,W,alpha,max_iterations,lamda=0):
        return self.linearRegression(X,Y,W,alpha,lamda,max_iterations)

    def validate(self,X,Y,W):
        return self.calculateCost(X,Y,W)

    def test(self,X,Y,W):
        return self.validate(X,Y,W)