import numpy as np
class gaussianNB():

  def fit(self,X_train,Y_train):   #computes the mean and std_dev of all types of classes in train
    self.mean= {}
    self.std = {}
    # split the data based on no. of classes
    self.classes =np.unique(Y_train)
    for x in self.classes:
      indices = np.where(Y_train==x)
      self.mean[x] = np.mean(X_train[indices[0]],axis=0)
      self.std[x] = np.std(X_train[indices[0]],axis=0) 

  def predict(self,X_test):
    y_pred=[]
    for i in X_test:
      y_pred_prob= {}
      for x in self.classes:
        y_pred_prob[x] = np.sum((i- self.mean[x])**2/ (2*(self.std[x])**2))
    y_pred.append(max(y_pred_prob,key=y_pred_prob.get))               # selecting the class which has max probability
    return y_pred
