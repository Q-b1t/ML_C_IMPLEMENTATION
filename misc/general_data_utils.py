"""
A collection of functions for creating/parsing/preprocessing data that may be 
shared across multiple algorithms for experimentation purposes.
"""
import numpy as np


def get_xor():
  X = np.zeros((200,2))
  X[:50] = np.random.random((50,2)) / 2 + 0.5
  X[50:100] = np.random.random((50,2)) / 2
  X[100:150] = np.random.random((50,2)) / 2 + np.array([[0,0.5]])
  X[150:] = np.random.random((50,2)) / 2 + np.array([[0.5,0]])
  Y = np.array([0]*100 + [1]*100)
  return X,Y

def get_donut():
  """
  distance from origin is radius + random normal
  angle theta is uniformly distributed between (0, 2pi)
  """
  N = 200
  R_inner = 5
  R_outer = 10

  R1 = np.random.randn(N//2) + R_inner
  theta = 2*np.pi*np.random.random(N//2)
  X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

  R2 = np.random.randn(N//2) + R_outer
  theta = 2*np.pi*np.random.random(N//2)
  X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

  X = np.concatenate([ X_inner, X_outer ])
  Y = np.array([0]*(N//2) + [1]*(N//2))
  return X, Y