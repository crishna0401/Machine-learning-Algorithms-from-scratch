import  numpy as np 
from collections import defaultdict
# data is assumed to be numpy format (dont include labels in data)

class parzen_estimate():

  def __init__(self,window_type,length):
    self.length=length
    self.window_type = window_type

  def fit(self,x,X):
    count=defaultdict(int)
    for i in X:
      if self.window_type =='cube':
        condition = np.amax((np.abs(x[:-1]-i[:-1]))) <= self.length/2
      elif self.window_type =='gaussian':
        mean =x[:-1]
        sigma = self.length
        condition = ((1/sigma*np.sqrt(2*np.pi)) * np.exp(-1*(((x[:-1]-i[:-1])-mean)**2)/(2*sigma**2))).all()<= self.length/2

      if condition:
        count["{0}".format(i[-1])]+=1
    # print(count)
    return count; 

  def predict(self,test_data,X):
    pred=[]
    n = len(test_data)
    for x in test_data:
      dict1 = self.fit(x,X)

      if bool(dict1):
        max_key = max(dict1, key=dict1.get)
        # print(max_key)
      else:
        max_key = -1
      pred.append(int(max_key))
    return pred
      
  # y_test and y_pred should be of the from 1D array or list    
  def accuracy(self,y_test,y_pred):
    count =0
    for i in range(len(y_pred)):
      if y_pred[i]==y_test[i]:
        count+=1
    accuracy = count*100/len(y_pred)
    return accuracy
