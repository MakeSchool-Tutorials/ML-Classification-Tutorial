---
title: "Support Vector Classifiers"
slug: svm-classifier
---

## Section 2: Support Vector Machine Classifiers

Most of our time now has been spent with simple regression algorithms. 

Previously, we've only looked at how we can take regression models, identify data distributions that sync with our regression model, and use the model to classify our data.

In this example, we'll be talking about a model that does this at a higher level of abstraction, using some advanced linear mathematics. 

We'll be looking at the **Support Vector Machine**. 

So far, we've only been playing with the Iris dataset in SciKit-Learn. We'll continue with using it to convey why SVMs are so powerful. 

Let's get our data and model! 

```
from sklearn.datasets import load_iris
from sklearn.svm import SVC
```

You're probably wondering why we import the model `SVC()` and why not `SVM()`? 

Support Vector Machines are an entire subgroup of machine learning models used for much more than data classification. So for our intents and purposes, we want to specify that we only want to do data classification with our data.

Hence, why we import our *Support Vector Machine Classifier Algorithm*, or our `SVC()`. 

Now, let's get a closer look at our data. 

```
iris_data = load_iris()
print(iris_data.DESCR)
```

We're intimately familiar with the Iris data at this point, but as a quick review - we should already know that our classes for our data fall into three major classes: _Iris-Setosa_, _Iris-Versicolor_, and _Iris-Virginica_. 

Let's get a better look at our data using our familiar methods: `train_test_split()`. 

```
X, y = pd.DataFrame(data=iris.data, columns=iris.feature_names), pd.DataFrame(data=iris.target, columns=["iris_type"])
X.head()
```

Data looks good. Four major features, covering our width and length parameters for the Iris petals and sepals. And what of our targets?

```
y.head()
```

Data still looks good! Inclusive integer data ranging between 0 and 2 for each of the Iris classes. 

Let's recall that we can use `train_test_split()` to adequately and randomly segment our X and y data into partitions that we can use for improved model testing and fitness estimation.

```
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
y_train, y_test = np.ravel(y_train), np.ravel(y_test)
```

We should have now the partitions needed for adequately training and testing our Iris data. 

---

Now it's fun time.

Let's instantiate our SVM classifier.

Our SVM takes a few key arguments that we call *hyperparameters* that allow us to grossly and finely tune our model to fit our data best. 

The hyperparameters we'll concern ourselves with today include:
- **kernel**: ("linear", "rbf")
<br>Changes the _linear transformation function_ to fit around our data (decision boundary function).<br><br>

- **C**:      (0, inf)
<br>Controls trade-off between _smooth decision boundary_ and _minimizing training misclassification_.<br><br>

- **gamma**:  (0, inf)
<br>Kernel coefficient; controls _'fitness'_ of model to training data.

Don't get too confused on what all these means under the hood - you could spend a lifetime trying to do so. 

Instead, let's play around with our hyperparameters to find out what combination works best!

```
svc = SVC(kernel="linear", C=1.0, gamma="auto")
svc.fit(X_train, y_train)
```

The most important hyperparameter to understand here is our *kernel* argument, which basically tells the SVM how to partition our data.

In this case, we tell it to split our classes using linear functions, which we often call **lines**. 

Now that our simple SVM classifer is fitted to our data, let's make some predictions, shall we? 

```
y_pred = svc.predict(X_test)
print(y_pred)
```

Again, one can check that the shape of `y_pred` is what they expect it to be by comparing the tuple output of `y_pred.shape` with `y_test.shape`. If they are the exact same shape, then everything should be good!

Now, an array of predicted values is cool, but we want some explicit evaluation metrics. Particularly, we want to know our model's accuracy.

Let's get some accuracy scores using the `.score()` method. 

```
svc.score(X_test, y_test)
```

Not too shabby. 

**What other ways can we tune our model to better fit and predict on our data**? 

---

By the way, in case you're curious about what the SVM is doing under the hood... well, it's easier to simply *show* you!

Using the power of MatPlotLib, we can visualize the partition lines (boundary functions) that our classifier is generating. 

You certainly don't have to know how to write functions like the following - simply run the following function to see the visualized boundary functions!

```
def svc_visualized(iris, kernel="linear", C=1.0, gamma="auto"):
    X, y = iris.data, iris.target

    clf_svc_iris = SVC(kernel=kernel, C=C, gamma=gamma)
    clf_svc_iris.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    h = (x_max / x_min) / 100

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    plt.subplot(1, 1, 1)
    Z = clf_svc_iris.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.BuGn_r)
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.xlim(xx.min(), xx.max())
    plt.title("SVC (kernel='{}', C={}, gamma={})".format(kernel, C, gamma))
    
svc_visualized(iris_data)
```

You can even try changing up some of the values for *kernel*, *C*, and *gamma* to see how they change up our model. 

**Can you apply the concept of cross-validation and hyperparameter tuning to algorithmically determine the best set of hyperparameters to use**?

