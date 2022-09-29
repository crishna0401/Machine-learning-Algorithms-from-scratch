import numpy as np
import math

class FLDA():
  def transform(self,x):
    x=(x-np.mean(x,axis=0))/np.std(x,axis=0)
    # print(x.shape)
    return x

  def sigmoid(self,w,x,b):
    z=np.dot(w,x)+b
    if z < 0:
      return 1 - 1/(1 + math.exp(z))
    else:
      return 1/(1 + math.exp(-z))

  def split(self,x,y):
    x_0=[]
    x_1=[]
    for i in range(len(x)):
      if(y[i]==0):
        x_0.append(x[i])
      else:
        x_1.append(x[i])
    return np.array(x_0),np.array(x_1)

  def pred(self,w,x,b):
    z=self.sigmoid(w,x,b)
    return z

  def accuracy(self,y_test,y_pred,thresh):
    y_pred=np.array(y_pred)
    y_pred = np.where(y_pred>=thresh,1,0)
    count =0
    for i in range(len(y_pred)):
      if y_pred[i]==y_test[i]:
        count+=1
    accuracy = count*100/len(y_pred)
    return accuracy
  
  def fit(self,x_train,y_train,x_val,y_val):
    self.x_train=self.transform(x_train)
    self.y_train=y_train
    self.x_val=self.transform(x_val)
    self.y_val=y_val
    self.x0,self.x1=self.split(self.x_train,self.y_train)
    self.m0=np.mean(self.x0,axis=0)
    self.m1=np.mean(self.x1,axis=0)
    self.s0=np.dot((self.x0-self.m0).T,(self.x0-self.m0))
    self.s1=np.dot((self.x1-self.m1).T,(self.x1-self.m1))
    self.sw = self.s0 + self.s1
    self.w = np.dot(np.linalg.inv(self.sw),(self.m1-self.m0))
    max_acc=0
    b1=-np.inf
    for b in range(-31,31):
      y_pred=[]
      for i in range(len(self.x_val)):
        y_pred.append(self.pred(self.w,self.x_val[i],b/10))
      acc=self.accuracy(self.y_val,y_pred,thresh=0.5)
      if(acc>max_acc):
        max_acc=acc
        b1=b/10
    # print(b1,max_acc)
    self.b = b1
    
  def predict(self,x_test):
    x_test=self.transform(x_test)
    y_pred=[]
    for i in range(len(x_test)):
      y_pred.append(self.pred(self.w,x_test[i],self.b))
    return y_pred

    
