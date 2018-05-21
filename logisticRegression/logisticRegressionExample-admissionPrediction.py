import numpy as np
import pandas as pd
from datetime import datetime

class RegressionModel:
    def __init__(self, X):
        self.learningRate = 1e-2

    def hipotesis(self, X, tita):
        return 1/(1+np.e**(-X.dot(tita)))

    def updateWeights(self, X, y, tita):
        tita -= self.learningRate * self.gradient(X, y, tita)
        return tita

    def train(self, X, y, tita):
        self.updateWeights(X, y, tita)

    def costFunction(self, X, y, tita):
        m = len(X)
        return (1/m)*(-y.transpose().dot(np.log(self.hipotesis(X, tita)))-(1-y).transpose().dot(np.log(1-self.hipotesis(X, tita))))

    def gradient(self, X, y, tita):
        m = len(X)
        return (1/m)*(X.transpose().dot(self.hipotesis(X, tita)-y))

    def prediction(self, X, tita):
        return self.sigmoid(X, tita)
    
    def sigmoid(self, X, tita):
        return self.hipotesis(X, tita) >= 0.5

    def classificationRate(self, predictedValues, y):
        m = len(y)
        print('equals values {}'.format(np.sum(predictedValues == y)))
        print('total values {}'.format(m))
        return np.sum(predictedValues == y)/m * 100

def main():
    df = pd.read_csv('ex2data1.txt', sep=',')
    
    X = df.iloc[:, :2]
    y = df.iloc[:, 2:]
    
    X0 = pd.DataFrame(np.ones(99))
    X = X0.join(X)
    
    tita = pd.DataFrame(np.random.rand(3, 1))

    X = X.as_matrix()
    y = y.as_matrix()
    tita = tita.as_matrix()
    m = len(X)
    
    rm = RegressionModel(X)

    print('learningRate {}'.format(rm.learningRate))

    totalruns = 10**10
    currentRun = 0

    startTime = datetime.now()
    print('Started at: {}'.format(startTime))

    for i in range(totalruns):
        currentRun += 1
        print('Percentaje completed : {}'.format((currentRun/totalruns)*100), end='%\r')
        rm.train(X, y, tita)

    endTime = datetime.now()
    print('Finished at: {}'.format(endTime))

    print('classification rate:')
    print(rm.classificationRate(rm.prediction(X, tita), y))

if __name__ == "__main__":
    main()
