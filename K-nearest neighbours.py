import numpy as np
from statistics import mode

class KNN():
  def __init__(self,k,p):
    self.k= k
    self.p=p

  def fit(self,X_train,Y_train):
    # normalise X_train
    self.X_train = (X_train-X_train.mean(axis=0))/X_train.std(axis=0)
    self.Y_train = Y_train

  def predict(self,X_test):
    # normalise X_test
    X_test = (X_test-X_test.mean(axis=0))/X_test.std(axis=0)
    Y_pred=[]
    for data in X_test:
      distances = np.sum(np.abs(self.X_train - data)**self.p,axis=1)**(1/self.p) 
      distances = distances.tolist()
      indices = sorted(range(len(distances)), key = lambda sub: distances[sub])[:self.k]
      neighbours = [self.Y_train[i] for i in indices]      
      Y_pred.append(mode(neighbours))
    return Y_pred
