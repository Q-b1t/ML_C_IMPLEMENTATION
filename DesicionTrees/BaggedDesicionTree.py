import numpy as np
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor


class BaggedTreeRegressor:
  def __init__(self,B):
    """
    B: Number of bagging rounds
    """
    self.B = B

  def fit(self,X,Y):
    """
    Trains a bagged decision tree regresor
    """
    N = len(X) # get the size of the dataset
    self.models = list()

    # iterate over the number of bagging rounds training one model per round
    for _ in range(self.B):
      # get a random subsampled dataset
      idx = np.random.choice(N,size=N,replace=True) # the samples are replaced once taken into the dataset
      Xb, Yb = X[idx],Y[idx]

      # train the model
      model =   DecisionTreeRegressor()
      model.fit(Xb, Yb)
      self.models.append(model)

  def predict(self,X):
    predictions = np.zeros(len(X))
    for model in self.models:
      predictions += model.predict(X)
    return predictions / self.B

  def score(self,X,Y):
     d1 = Y - self.predict(X)
     d2 = Y - Y.mean()
     return 1 - d1.dot(d1) / d2.dot(d2)



class BaggedTreeClassifier:
  def __init__(self,B):
    self.B = B

  def fit(self,X,Y):
    # get the dataset size
    N = len(X)
    self.models = list()

    for _ in range(self.B):
      # get random subsampled dataset
      idx = np.random.choice(N,size=N,replace=True)
      Xb,Yb = X[idx],Y[idx]

      # train a model
      model = DecisionTreeClassifier(max_depth = 2) # to have a smother decision boundary
      model.fit(Xb,Yb)
      self.models.append(model)

  def predict(self,X):
    predictions = np.zeros(len(X))
    for model in self.models:
      predictions += model.predict(X)
    return np.round(predictions / self.B)

  def score(self,X,Y):
    P = self.predict(X)
    return np.mean(Y==P)