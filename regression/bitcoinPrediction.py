import numpy as np
import pandas as pd
from datetime import datetime

class RegressionModel:

    def __init__(self, X):
        self.learningRate = 0.0000000000000000000000000000000000000000000000001
        self.tita0 = np.random.rand()
        self.tita1 = np.random.rand()
        self.size = X.count()

    def hipotesis(self, x):
        print('hipotesis {}'.format(self.tita0 + self.tita1 * x))
        return self.tita0 + self.tita1 * x

    def updateWeights(self,x, y):
        print('X value {}'.format(x))
        print('Y value {}'.format(y))
        print('tita0 {}'.format(self.tita0))
        print('tita1 {}'.format(self.tita1))
        newValueTita0 = (1/self.size) * (self.hipotesis(x) - y) * x
        newValueTita1 = (1/self.size) * (self.hipotesis(x) - y)
        print('newValueTita0 {}'.format(newValueTita0))
        print('newValueTita1 {}'.format(newValueTita1))
        self.tita0 = self.tita0 + self.learningRate *  newValueTita0
        self.tita1 = self.tita1 + self.learningRate *  newValueTita1

    def train(self, X, Y):
        for i in range(self.size):
            print("cost function: {}".format(self.costFunction(self.hipotesis(X[i]), Y[i])))
            self.updateWeights(X[i], Y[i])

    def costFunction(self, eval, y):
        return (1/2*self.size) * (eval - y)**2

def main():
    df = pd.read_csv('all_currencies.csv')
    btcValueByDate = df.loc[df['Symbol']=='BTC'][['Date', 'High']]
    X = pd.to_datetime(btcValueByDate['Date']).map(datetime.toordinal)[:500]
    Y = pd.to_numeric(btcValueByDate['High'])[:500]
    Xm = X.as_matrix()
    Ym = Y.as_matrix()
    rm = RegressionModel(X)

    print('tita0 {}'.format(rm.tita0))
    print('tita1 {}'.format(rm.tita1))
    print('size {}'.format(rm.size))
    print('learningRate {}'.format(rm.learningRate))

    for i in range(1000):
        rm.train(Xm, Ym)

    print(datetime.fromordinal(Xm[499]))
    # print(rm.costFunction(rm.hipotesis(Xm[499]), Ym[499]))
    # rm.updateWeights(Xm[499], Ym[499])
    print(rm.hipotesis(Xm[499]))


if __name__ == "__main__":
    main()
