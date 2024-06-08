import numpy as np
from scipy.stats import multivariate_normal as mvn

class GaussianNaiveBayes:
  def __init__(self,smoothing = 1e-3):
    self.smoothing = smoothing
  
  def fit(self,X,Y):
    K = len(set(Y)) # get the number of distinct classes
    N,D = X.shape # get the number of samples and features respectively

    self.log_priors = np.zeros(K) # placeholder for storing the prior probabilities p(y = k)
    self.means = np.zeros((K,D)) # placeholder for storing the sample means (inputs of size D for each class k)
    self.variances = np.zeros((K,D)) # store the covariance matrix (assuming independence between features, we only require the diagonal)

    for k in range(K):
      # compute p(y = k)
      self.log_priors[k] = np.log(len(Y[Y == k])) - np.log(N)

      # compute the likelyhood 
      X_k = X[Y == k] # get a subset with only samples corresponding to k
      self.means[k] = X_k.mean(axis = 0)
      self.variances[k] = X_k.var(axis = 0) + self.smoothing

  def score(self, X, Y):
    P = self.predict(X)
    return np.mean(P == Y)

  def predict(self, X):
    N,D = X.shape # get the number of features and samples
    K = len(self.log_priors) # get the number of classes
    P = np.zeros((N,K)) # # placeholder for placing the posterior probabilities p(y = x)

    for k, pr, m, v in zip(range(K), self.log_priors, self.means, self.variances):
      P[:, k] = mvn.logpdf(X, mean=m, cov=v) + pr
    return np.argmax(P, axis=1)
