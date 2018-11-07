---
title: "Linear Regression Classifiers"
slug: lin-reg-classifier
---

## Section 1: The *Linear Regression* Classifier

In our first tutorial, we'll be discussing one of the simplest classifier models in all of machine learning: the *linear regression* model. 

Now, those of you with perhaps some prior machine learning experience may be wondering... why on Earth would a regression model be used for classification? 

Brilliant question!

As we should know, a regression model attempts to fit data as best it can to a single function, with hopes of being able to extrapolate a mathematical pattern to describe data that doesn't exist or lies outside of our access. 

These regression models can be incredibly advanced and span the realms of multivariate engineering mathematics, but generally in programming we reference the simpler ones. 

A linear regression model is where we attempt to fit our data to a single line on a graph. 

We can extrapolate a regression model to be able to classify our data simply by making the line of best fit our split (decision boundary).

---

Let's get started by running our basic imports: one for our *dataset* and one for our *model*. 

```
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression()
```

In this example, we're going to use the Iris dataset - a classic dataset for machine learning classification. 

For all basic examples in our tutorials, we'll be using datasets we could import from SciKit-Learn. 

```
iris_data = load_iris()
print(iris_data.DESCR)
```

That `print()` statement allows us to see descriptive information on the dataset. 

The Iris data contains four main features describing the length and width of Iris petals and sepals. 

Our target over this dataset is the type of Iris flower. The dataset contains three major target classes: _Setosa_, _Versicolor_, and _Virginica_. 

---

Now that we have our data imported, let's instantiate our dataset and target vector as `X` and `y`. We'll create these variables as Pandas DataFrame objects. 

After instantiating our data-containing objects, let's peek at the dataset to examine our features. 

```
X, y = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names), pd.DataFrame(data=iris_data.target, columns=["iris_type"])
X.head()
```

As we can see, we do indeed have our four main features: _sepal length (cm)_, _sepal width (cm)_, _petal length (cm)_, and _petal width (cm)_. 

Each feature in our dataset contains continuous data, which is essential for appropraite regression calculation.

We can peek at our target vector as well, just to see if our classes are consistent with what we expect. 

```
y.head()
```

We're good! It looks like our target vector contains inclusive integer values between 0 and 2, representing our three discrete classes!

Now that we know our data is there, let's partition our data a little bit to optimize our model training and testing. 

We'll be using a module called `train_test_split()` that allows us to randomly partition our data. This will make more sense as we put it in practice.

```
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
y_train, y_test = np.ravel(y_train), np.ravel(y_test)
```

That final command is needed to restructure our y-data in order to effectively fit and predict the model on it. 

---

Speaking of which, now that we've effectively partitioned our training and testing datasets, let's put them to use. 

Our Data Modeling Process entails training our model on our training datasets and sending a testing set of feature data to our model to predict. 

By comparing the predicted y-values with our true test y-values, we can ascertain our model's accuracy. 

One way to do this is by manually iterating across our predicted y-values (`y_pred`) and true test y-values (`y_test`) and checking which values are equivalent.

However, we can do this by simply calling `.score()` on our machine learning model. 

Let's try this process out now by instantiating our model.

```
linreg = LinearRegression()
linreg.fit(X_train, y_train)
```

Here, we've just *instantiated* our machine learning model (our **linear regression** model) by assigning an empty model to a variable.

Then, we *fitted* the model to our training dataset. Here, it learns the approximate relationship between the X and y datasets. 

Through learning the relationship between the X-y training data, we're hoping that the model can approximately determine the y-value given a new X-value. 

In our example, rather than sending it new data manually, we'll actually send the model our `X_test` data, since it hasn't seen it before. 

```
y_pred = linreg.predict(X_test)
```

Generally, we assign the name of the output array of a `.predict()` model command to be called `y_pred`, or the predicted y-values array.

By printing out and seeing the values across `y_pred`, we see what our model *thinks* should be the correct target labels for the corresponding `X_test` values.

A quick sanity check we can do to assure that our data is what we think it is is to call `.shape` on `y_pred` and `y_test`.

If the shape of both the predicted y-values array and the true test y-values array are consistent with one another, then we can assume that our model worked somewhat effectively.

---

The next step is to simply determine the model's accuracy. 

We can do this by calling `.score()` on our model and sending it our test data. 

```
linreg.score(X_test, y_test)
```

One common mistake many people make is to send `y_pred` rather than `X_test` as the first argument to the scoring method. 

The scoring method actually does the prediction step under the hood, so it requires that you send it both testing datasets. 

---

Now that we have our score, let's think about it. Generally, we want our score to beat a baseline score, which effectively can be presented as the chance of getting a single class label randomly.

In this case, since we have three explicit class labels, we technically have a 33.33% baseline probability to correctly assign any label.

So let's apply that to our obtained score. 

**Did our linear classification score beat the relative baseline**? 
