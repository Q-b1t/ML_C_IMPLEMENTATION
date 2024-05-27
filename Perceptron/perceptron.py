import numpy as np

class Perceptron:
  def fit(self,X,Y,learning_rate = 1.0,epochs=1000):
    # initialize random weights
    D = X.shape[1]
    self.w = np.random.randn(D)
    self.b = 0

    N = len(Y)
    costs = list()

    for epoch in range(epochs):
      # determine whether there are some missclassified samples
      Y_hat = self.predict(X)
      # incorrect data
      incorrect = np.nonzero(Y != Y_hat)[0]

      if len(incorrect) == 0:
        break

      # choose a random incorrect sample
      i = np.random.choice(incorrect)
      self.w += learning_rate * Y[i] * X[i]
      self.b += learning_rate * Y [i]

      # cost is incorrect rate
      c = len(incorrect) / float(N)
      costs.append(c)
    print(f"Final w: {self.w} | Final b: {self.b} | Epochs: {epoch +1}/{epochs}")
    return costs

  def predict(self,X):
    return np.sign(X.dot(self.w) + self.b)

  def score(self,X,Y):
    P = self.predict(X)
    return np.mean(P == Y)