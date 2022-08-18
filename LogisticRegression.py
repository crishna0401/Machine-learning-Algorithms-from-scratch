import numpy as np

class LogisticRegression():

  def __init__(self,lr,tolerance,max_iter):
    self.lr = lr
    self.tolerance = tolerance
    self.max_iter = max_iter

  def sigmoid(self,x):
    return 1/(1+np.exp(-x))

  def fit(self,X,y):
    # Append ones column so that we account for bias term 
    # Now the number of dimensions of X increase by 1
    X = np.hstack((X,np.ones(X.shape[0]).reshape(X.shape[0],1)))

    # Intialize parameters using normal distribution
    self.w = np.random.randn(X.shape[1],1)
    
    # predict the output vector
    y_pred = self.sigmoid(X@self.w)

    # check L2 loss between predicted and actual vectors
    loss = -np.sum(np.multiply(y,np.log(y_pred)) + np.multiply((1-y),np.log(1-y_pred)))

    ###### Run a simple gradient descent loop for minimising loss #######
    while loss>self.tolerance and self.max_iter:
      gradient = -(X.T @ (y-y_pred))

      # update the parameters
      self.w = self.w - self.lr*(gradient)

      # with the updated parameters predict the output again and update loss
      y_pred = self.sigmoid(X@self.w)
      loss = -np.sum(np.multiply(y,np.log(y_pred)) + np.multiply((1-y),np.log(1-y_pred)))
      self.max_iter-=1

    return loss,self.w

  def predict_prob(self,X):
    X = np.hstack((X,np.ones(X.shape[0]).reshape(X.shape[0],1)))
    # y = X.W is the linear form assumed and this function returns X.W which is prediction vector
    return self.sigmoid(X@self.w)

  def predict(self,X,threshold=0.5):
    X = np.hstack((X,np.ones(X.shape[0]).reshape(X.shape[0],1)))
    pred = self.sigmoid(X@self.w)
    pred [pred>=threshold] = 1
    pred [pred<threshold] =0 
    return pred
