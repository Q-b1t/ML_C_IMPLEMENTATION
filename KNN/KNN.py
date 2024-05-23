from future.utils import iteritems
import numpy as np
from sortedcontainers import SortedList
from tqdm import tqdm

# k-mean nearest neighboors class
class KNN(object):
  def __init__(self,K):
    super().__init__()
    self.K = K

  def fit(self,X,y):
    self.X,self.y = X,y # lazy classifier (the training only stores the data)

  def predict(self,X): # receives a sample or batch of sampes and makes a preidciton
    y = np.zeros(len(X)) # create an array to store the prediction
    for i,x in tqdm(enumerate(X)): # iterates over the batch of samples to predict
      sorted_list = SortedList() # create a sorted list per sample to store (distance, class) pairs for all the training samples
      for j,x_d in enumerate(self.X): # iterates over the dataset
        # calculate the distance (euclidean)
        diff = x-x_d
        distance = diff.dot(diff)
        if len(sorted_list) < self.K: # if we have not reached the k neigboors, we can add indiscriminately
          sorted_list.add(
              (distance,self.y[j])
          )
        else: # if we have already supassed the number of nearest neighboors, compare to see if the distance is smaller
          if distance < sorted_list[-1][0]:
            del sorted_list[-1]
            sorted_list.add(
                (distance,self.y[j])
            )

      # votes
      votes = dict()
      for _,v in sorted_list:
        votes[v] = votes.get(v,0)+1

      max_votes,max_votes_class = 0,-1

      for v,count in iteritems(votes):
        if count > max_votes:
          max_votes = count
          max_votes_class = v
      y[i] = max_votes_class
    return y

  def score(self, X, Y):
    P = self.predict(X)
    return np.mean(P == Y)