import numpy as np
import pandas as pd


# get the data
def get_data(path = "train.csv",limit = None):
  df = pd.read_csv(path)
  data = df.values
  np.random.shuffle(data)
  X = data[:, 1:] / 255.0 # data is from 0..255
  Y = data[:, 0]
  if limit is not None:
      X, Y = X[:limit], Y[:limit]
  return X, Y