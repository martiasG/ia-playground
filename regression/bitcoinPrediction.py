import numpy as np
import pandas as pd
from datetime import datetime

class RegressionModel:

    def __init__(self, X):
        self.learningRate = 1e-3
        # self.tita0 = np.random.rand()
        # self.tita1 = np.random.rand()
        self.tita = np.random.rand(2)
        self.size = X.count()

    def hipotesis(self, X):
        # Slow
        # return self.tita0 + self.tita1 * X
        return X.dot(self.tita.transpose())

    def updateWeights(self,x, y, predictValue):
        # Slow version
        # error = (1/self.size) * (predictValue - y)
        # newValueTita0 = (error * x).sum()
        # newValueTita1 = error.sum()
        # self.tita0 += self.learningRate *  newValueTita0
        # self.tita1 += self.learningRate *  newValueTita1
        # Fast version vectorial calculus
        # gradient descent
        newTitaValues = (1/self.size) * (predictValue - y).transpose().dot(x)
        self.tita -= self.learningRate * newTitaValues
        # Normal ecuation
        # O(n**3) cost
        # good for n < 10**5
        # xt = x.transpose()
        # newTitaValues = np.linalg.inv(xt.dot(x)).dot(xt).dot(y)
        # no need to choose a learning rate
        # self.tita = newTitaValues


    def train(self, X, Y):
        #Slow
        # for i in range(self.size):
        #     predictValue = self.hipotesis(X[i])
        #     print("cost function: {}".format(self.costFunction(predictValue, Y[i])))
        #     self.updateWeights(X[i], Y[i], predictValue)
        #Fast
        self.updateWeights(X, Y, self.hipotesis(X))


    def costFunction(self, eval, y):
        return (1/2*self.size) * (eval - y)**2

def main():
    df = pd.read_csv('all_currencies.csv')
    btcValueByDate = df.loc[df['Symbol']=='BTC'][['Date', 'Close']]
    X = pd.to_datetime(btcValueByDate['Date']).map(datetime.toordinal)
    Y = pd.to_numeric(btcValueByDate['Close'])
    Xm = X.as_matrix()
    uix = Xm.sum()/X.count()
    maxX = Xm.max()
    print('max value: {}'.format(maxX))
    minX = Xm.min()
    print('min value: {}'.format(minX))
    print('average value x: {}'.format(uix))
    normalizedx = (Xm - uix) / (maxX - minX)
    normalizedXpluss1 = np.column_stack((np.ones(len(normalizedx)), normalizedx))

    Ym = Y.as_matrix()
    uiy = Ym.sum()/Y.count()
    normalizedy = (Ym - uiy) / (Ym.max() - Ym.min())

    rm = RegressionModel(X)

    print('tita0 {}'.format(rm.tita[0]))
    print('tita1 {}'.format(rm.tita[1]))
    print('size {}'.format(rm.size))
    print('learningRate {}'.format(rm.learningRate))
    print('y shape {}'.format(Ym.shape))

    totalruns = 10**5
    # for normal equation
    # totalruns = 1
    currentRun = 0

    startTime = datetime.now()
    print('Started at: {}'.format(startTime))

    for i in range(totalruns):
        currentRun += 1
        print('Percentaje completed : {}'.format((currentRun/totalruns)*100), end='%\r')
        # rm.train(Xm, Ym)
        rm.train(normalizedXpluss1, normalizedy)

    # print('Predict value for: {} = {}'.format(datetime.fromordinal(Xm[499]), rm.hipotesis(Xm[499])))
    print('Predict value for: {}'.format(rm.hipotesis((normalizedXpluss1[700]))*(Ym.max() - Ym.min())+uiy))
    # print('Real value for: {} = {}'.format(datetime.fromordinal(Xm[499]), Ym[499]))
    print('Real value: {}'.format((normalizedy[700])*(Ym.max() - Ym.min())+uiy))
    endTime = datetime.now()
    print('Finished at: {}'.format(endTime))

if __name__ == "__main__":
    main()
