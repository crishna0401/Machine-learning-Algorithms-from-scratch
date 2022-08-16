import numpy as np

class LinearRegression():

  def __init__(self,lr,tolerance):
    self.lr = lr
    self.tolerance = tolerance

  def fit(self,X,y):
    # Append ones column so that we account for bias term 
    # Now the number of dimensions of X increase by 1
    X = np.hstack((X,np.ones(X.shape[0]).reshape(X.shape[0],1)))

    _,d = X.shape

    # Intialize parameters using normal distribution
    self.w = np.random.randn(d,1)
    
    # predict the output vector
    y_pred = self.predict(X)

    # check L2 loss between predicted and actual vectors
    loss = np.linalg.norm(y-y_pred)

    ###### Run a simple gradient descent loop for minimising loss #######
    while loss>self.tolerance:
      gradient = -2*(X.T @ (y-y_pred))

      # update the parameters
      self.w = self.w - self.lr*(gradient)

      # with the updated parameters predict the output again and update loss
      y_pred = self.predict(X)
      loss = np.linalg.norm(y-y_pred)

    return loss,self.w

  def predict(self,X):
    # y = X.W is the linear form assumed and this function returns X.W which is prediction vector
    return X@self.w
