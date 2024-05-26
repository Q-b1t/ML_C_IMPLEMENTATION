import numpy as np

class TreeNode:
  def __init__(self,depth = 1,max_depth = None):
    print(f"[~] Depth: {depth}")
    self.depth = depth
    self.max_depth = max_depth
    # validate the parameters
    if self.max_depth is not None and self.max_depth < self.depth:
      raise Exception("[-]: depth > max_depth")

  def fit(self,X,Y):
    if(len(Y) == 1 or len(set(Y)) == 1):
      # base case there there is either all the samples of the same class or a single sample
      self.col = None
      self.split = None
      self.left = None
      self.right = None
      self.prediction = Y[0] # predict the only class available

    else:
      D = X.shape[1] # get the number of attributes
      cols = range(D) # get the number of attributes.
      # set up the placeholders
      max_ig,best_col,best_split = 0,None,None
      for col in cols:
        ig,split = self.find_split(X,Y,col) # find the information gain and the most optimal split for the column in question
        if ig > max_ig:
          max_ig = ig
          best_col = col
          best_split = split

      if max_ig == 0: # if after going through all the features, the information gain continues to be zero,there is nothing else we can do
        self.col = None
        self.split = None
        self.left = None
        self.right = None
        self.prediction = np.round(Y.mean())

      else:
        self.col = best_col
        self.split = best_split

        if self.depth == self.max_depth: # stopping condition for the recursion
          self.left = None
          self.right = None
          self.prediction = [
              np.round(Y[X[:,best_col] < self.split].mean()),
              np.round(Y[X[:,best_col] >= self.split].mean())
          ]
        else:
          # recursion based on the split

          # left node
          left_idx = (X[:,best_col] < best_split)
          # get two datasets based on the new split
          X_left,Y_left = X[left_idx],Y[left_idx]
          self.left = TreeNode(self.depth + 1,self.max_depth)
          self.left.fit(X_left,Y_left)

          # right side
          right_idx = (X[:,best_col] >= best_split)
          X_right,Y_right = X[right_idx],Y[right_idx]
          self.right = TreeNode(self.depth + 1,self.max_depth)
          self.right.fit(X_right,Y_right)

  def find_split(self,X,Y,col):
    # sort the samples and the labels
    x_values = X[:, col]
    sort_idx = np.argsort(x_values) # get index mapping to the sorted array
    x_values = x_values[sort_idx]
    y_values = Y[sort_idx]


    # nonzero() gives us indices where arg is true
    boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0] # find the indexes where there is a change of value in the labels (potential split points)
    best_split = None
    max_ig = 0

    for b in boundaries: # iterative search of the best information gain using all the potential boundaries
      split = (x_values[b] + x_values[b+1]) / 2
      ig = self.information_gain(x_values,y_values,split)
      if ig > max_ig:
        max_ig = ig
        best_split = split
    return max_ig,best_split


  def entropy(self,y):
    # assume binary entropy function: n e {0:1}
    N = len(y)
    s1 = (y == 1).sum() # obtain the number of samples equal to 1
    if 0 == s1 or N == s1: # if there is only one class the entropy is zero
      return 0

    p_1 = float(s1) / N # normalize the data to be in the range [0:1]
    p_0 = 1 - p_1  # the number of zeros is the total - the number of ones
    return - p_0 * np.log2(p_0) - p_1 * np.log2(p_1) # compute the entropy


  def information_gain(self, x, y, split):
    # create two different subsets of labels based on the given split
    y_0 = y[x < split]
    y_1 = y[x >= split]
    N = len(y)
    y0_len = len(y_0)

    # the information gain will be 0 if there is only one sample or all the labels correspond to the same class (it is already a deterministic sampling)
    if y0_len == 0 or y0_len == N:
      return 0

    # compute the information gain
    p_0 = float(len(y_0)) / N
    p_1 = 1 - p_0 #float(len(y1)) / N
    return self.entropy(y) - p_0 * self.entropy(y_0) - p_1 * self.entropy(y_1)


  def predict_one(self, x):
    # use "is not None" because 0 means False
    if self.col is not None and self.split is not None:
      feature = x[self.col]
      if feature < self.split:
        if self.left:
          p = self.left.predict_one(x)
        else:
          p = self.prediction[0]
      else:
        if self.right:
          p = self.right.predict_one(x)
        else:
          p = self.prediction[1]
    else:
        # corresponds to having only 1 prediction
      p = self.prediction
    return p

  def predict(self,X):
    N = len(X)
    p = np.zeros(N)
    for i in range(N):
      p[i] = self.predict_one(X[i])
    return p

class DesicionTree:
  def __init__(self,max_depth=None):
    self.max_depth = max_depth

  def fit(self,X,Y):
    self.root = TreeNode(
        max_depth=self.max_depth
    )

    self.root.fit(X,Y)

  def predict(self,X):
    return self.root.predict(X)

  def score(self,X,Y):
    P = self.predict(X)
    return np.mean(P == Y)