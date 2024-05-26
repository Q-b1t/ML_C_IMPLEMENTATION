# Machine learning algorithms
Implementations of machine learning (not Deep Learning nor other paradigm) algorithms.

# import training data
Import pertinent datasets and utilities for my own reference:
## get pertinent utilities 
```
!rm -r ML_C_IMPLEMENTATION utils
!git clone https://github.com/Q-b1t/ML_C_IMPLEMENTATION
!mkdir utils
!cp ML_C_IMPLEMENTATION/misc/*.py utils
```
## get some training data
this is an example for the mnist dataset, which i have used a lot recently.
```
!rm -r G.A.I_MISCELLANEUS_RESOURCES mnist_data
!git clone https://github.com/Q-b1t/G.A.I_MISCELLANEUS_RESOURCES.git
!mkdir mnist_data
!unzip /content/G.A.I_MISCELLANEUS_RESOURCES/digit_recognizer/digit-recognizer.zip -d mnist_data
```