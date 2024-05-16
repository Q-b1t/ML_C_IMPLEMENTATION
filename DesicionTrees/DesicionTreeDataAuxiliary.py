from future.utils import iteritems
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def get_columns(type="regression"):
  columns = {
    "numerical": {
      "numerical": [
        'crim', # numerical
        'zn', # numerical
        'nonretail', # numerical
        'nox', # numerical
        'rooms', # numerical
        'age', # numerical
        'dis', # numerical
        'rad', # numerical
        'tax', # numerical
        'ptratio', # numerical
        'b', # numerical
        'lstat', # numerical
        ],
      "no_transform":['river']
    },
    "classification":{
      "numerical":(),
      "categorical":np.arange(22) + 1
    }
  }
  return columns[type]



# class to transform the data
class DataTransformerRegressor:
  def fit(self,df,cols):
    numerical_cols = cols["numerical"]
    self.scalers = dict()
    for col in numerical_cols:
      scaler = StandardScaler()
      scaler.fit(df[col].values.reshape(-1,1))
      self.scalers[col] = scaler

  def transform(self,df,cols):
    numerical_cols,no_transform_cols = cols["numerical"],cols["no_transform"]
    N,_ = df.shape
    D = len(numerical_cols) + len(no_transform_cols)
    X = np.zeros((N,D)) # sample_num X feature_num

    i = 0

    for col,scaler in iteritems(self.scalers):
      X[:,i] = scaler.transform(df[col].values.reshape(-1, 1)).flatten()
      i += 1
    for col in no_transform_cols:
      X[:,i] = df[col]
      i += 1
    return X

  def fit_transform(self, df):
    self.fit(df)
    return self.transform(df)








class DataTransformerClassifier:
  def fit(self, df,cols):
    numerical_columns,categorical_columns = cols["numerical"],cols["categorical"]
    self.labelEncoders = dict()
    self.scalers = dict()
    for col in numerical_columns:
      scaler = StandardScaler()
      scaler.fit(df[col].reshape(-1, 1))
      self.scalers[col] = scaler

    for col in categorical_columns:
      encoder = LabelEncoder()
      # in case the train set does not have 'missing' value but test set does
      values = df[col].tolist()
      values.append('missing')
      encoder.fit(values)
      self.labelEncoders[col] = encoder

    # find dimensionality
    self.D = len(numerical_columns)
    for col, encoder in iteritems(self.labelEncoders):
      self.D += len(encoder.classes_)
    print("dimensionality:", self.D)



  def transform(self, df):
    N, _ = df.shape
    X = np.zeros((N, self.D))
    i = 0
    for col, scaler in iteritems(self.scalers):
      X[:,i] = scaler.transform(df[col].values.reshape(-1, 1)).flatten()
      i += 1











def preprocess_housing_data(data_path="housing.data"):
  df = pd.read_csv('housing.data', header=None, delim_whitespace=True)
  df.columns = [
    'crim', # numerical
    'zn', # numerical
    'nonretail', # numerical
    'river', # binary
    'nox', # numerical
    'rooms', # numerical
    'age', # numerical
    'dis', # numerical
    'rad', # numerical
    'tax', # numerical
    'ptratio', # numerical
    'b', # numerical
    'lstat', # numerical
    'medv', # numerical -- this is the target
  ]

  transformer = DataTransformerRegressor()
  # shuffle the data
  N = len(df)
  train_idx = np.random.choice(N, size=int(0.7*N), replace=False) # select random samples accounting for 70% of the training data
  test_idx = [i for i in range(N) if i not in train_idx] # use the remaining unselected samples as the test dataet
  # extract the data using the generated indices
  df_train = df.loc[train_idx]
  df_test = df.loc[test_idx]
  # preprocess the data using the custom transformer
  X_train = transformer.fit_transform(df_train)
  y_train = np.log(df_train['medv'].values)
  X_test = transformer.transform(df_test)
  y_test = np.log(df_test['medv'].values)
  return X_train, y_train, X_test, y_test


def replace_missing(df,cols):
  numerical_cols,categorical_cols = cols["numerical"],cols["categorical"]
  # standard method of replacement for numerical columns is median
  for col in numerical_cols:
    if np.any(df[col].isnull()):
      med = np.median(df[ col ][ df[col].notnull() ])
      df.loc[ df[col].isnull(), col ] = med

  # set a special value = 'missing'
  for col in categorical_cols:
    if np.any(df[col].isnull()):
      print(col)
      df.loc[ df[col].isnull(), col ] = 'missing'


def preprocess_mushrom_data(data_path = "agaricus-lepiota.data"):
  df = pd.read_csv(data_path, header=None)

  # replace label column: e/p --> 0/1
  # e = edible = 0, p = poisonous = 1
  df[0] = df.apply(lambda row: 0 if row[0] == 'e' else 1, axis=1)

  # check if there is missing data
  replace_missing(df)

  # transform the data
  transformer = DataTransformerClassifier()

  X = transformer.fit_transform(df)
  Y = df[0].values
  return X, Y