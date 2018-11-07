---
title: "Setup for Data Science"
slug: setup
---

## Tutorial: *DS 2.1 Classification Algorithms*

In this tutorial, we'll be exploring basic machine learning algorithms applied on to simple datasets with an emphasis on getting basic model functionality working before scaling up our model to fit more advanced data. 

We'll be exploring a range of algorithms including:
- Linear Regression Classifiers
- Logistic Regression Classifiers
- Support Vector Machines
- k-Nearest Neighbors Classifiers
- Decision Tree Classifiers
- Naive Bayes Classifiers

In order to ensure you'll be able to do classification tutorial sections appropriately, please ensure you can run the following cells in your Jupyter environment.

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
```

Many of these commands should be familiar to you. 

In this course, we'll be utilizing many familiar Data Exploration and Analysis libraries, including but not limited to:
- Pandas (*pd*)
- NumPy (*np*)
- MatPlotLib's PyPlot module (*plt*)
- Seaborn (*sns*)

When we talk about machine learning, we'll introduce a new and very powerful built-in Python module called *sklearn*, which allows us to do a variety of beginner, intermediate, and advanced tasks regarding supervised machine learning, particularly with classification.