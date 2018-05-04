import numpy as np
import pandas as pd
from datetime import datetime

class RegressionModel:

    def __init__(self, X):
        self.learningRate = 1e-7
        self.tita0 = np.random.rand()
        self.tita1 = np.random.rand()
        self.size = X.count()

    def hipotesis(self, X):
        #Slow
        # return self.tita0 + self.tita1 * x
        #Fast
        return self.tita0 + self.tita1 * X

    def updateWeights(self,x, y, predictValue):
        newValueTita0 = ((1/self.size) * (predictValue - y) * x).sum()
        newValueTita1 = ((1/self.size) * (predictValue - y)).sum()
        self.tita0 -= self.learningRate *  newValueTita0
        self.tita1 -= self.learningRate *  newValueTita1

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
    btcValueByDate = df.loc[df['Symbol']=='BTC'][['Date', 'High']]
    X = pd.to_datetime(btcValueByDate['Date']).map(datetime.toordinal)
    Y = pd.to_numeric(btcValueByDate['High'])
    Xm = X.as_matrix()
    Ym = Y.as_matrix()
    rm = RegressionModel(X)

    print('tita0 {}'.format(rm.tita0))
    print('tita1 {}'.format(rm.tita1))
    print('size {}'.format(rm.size))
    print('learningRate {}'.format(rm.learningRate))

    for i in range(10**10):
        rm.train(Xm, Ym)

    print('Predict value for: {} = {}'.format(datetime.fromordinal(Xm[499]), rm.hipotesis(Xm[499])))
    print('Real value for: {} = {}'.format(datetime.fromordinal(Xm[499]), Ym[499]))



if __name__ == "__main__":
    main()
