import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# define the class for the adaboost algorithm
class AdaBoost:
  def __init__(self,M):
    """
    M -> Number of base models (desicion trees)
    """
    self.M = M 

  def fit(self,X,Y):
    # instance placeholders to store the base models and their weights
    self.models,self.alphas = list(),list()

    # get number of samples and features from the training dataset
    N,_ = X.shape

    # instance the initial weights from a uniform distribution
    W = np.ones(N) / N

    for m in range(self.M):
      # instance a tree
      tree = DecisionTreeClassifier(max_depth=1)
      tree.fit(X,Y,sample_weight=W)
      # make the prediction
      P = tree.predict(X)

      # compute the error for the base model at hand
      err = W.dot(P != Y)
      
      # compute the weight for this base model
      alpha = 0.5 * (np.log(1-err) - np.log(err))

      # compute the exponential cost and normalize
      W = W * np.exp(-alpha * Y * P)
      W = W / W.sum() 

      # save the weights
      self.models.append(tree)
      self.alphas.append(alpha)

  def predict(self,X):
    # this is not like the sklearn API. the purpose is to get accuracy and exponential loss for plotting
    N,_ = X.shape
    FX = np.zeros(N)
    for alpha,tree in zip(self.alphas,self.models):
      FX += alpha * tree.predict(X)
    return np.sign(FX), FX 


  def score(self,X,Y):
    # the same as above
    P,FX = self.predict(X)
    L = np.exp(-Y * FX).mean()
    return np.mean(P == Y), L