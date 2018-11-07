---
title: "k-Nearest Neighbors Classifiers"
slug: kNN-classifier
---

## Section 4: The *k-Nearest Neighbors* Algorithm

We've talked about how regression lines of best fit can be used to classify data, and we've talked about how that concept can be expanded to create advanced geometric constructs to linearly separate our data in a hyperdimensional manner.

Let's get even simpler than that for this algorithm.

The **k-Nearest Neighbors** algorithm allows you to create your decision boundary for classification based on a simple criterion: *how many of my neighbors share my classification*? 

Indeed, kNN allows us to simply iterate over our data and check each data point's classification relationship with other data points close to it in proximity (feature values). 

Points that are nearer to data values with differing classes are similar to our support vectors (SVM), where they are weighed higher with creating our decision boundary.

Let's see how this works!

```
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
```

For this data analysis, we'll be exploring the Wine dataset in SciKit-Learn (you can also find it via UCI's Open Dataset). 

At this point, we definitely should know how to look at our dataset's features if it's an SciKit-Learn dataset, but let's review it anyway.

```
wine_data = load_wine()
print(wine_data.DESCR)
```

The printed information tells us most importantly that there are three major classes for our wine classification labels: *class_0* (0), *class_1* (1), and *class_2* (2).

Let's quickly peek at our data's head to see what we're dealing with. 

```
X, y = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names), pd.DataFrame(data=wine_data.target, columns=["wine_quality_type"])
X.head()
```

Hmm, looks interesting. 13 columns of all continuous data. And what about our targets? 

```
y.head()
```

Inclusive integer values between 0 and 2. Looks like everything checks out! 

Let's get to work, first by making use of our old friend *train_test_split*. 

```
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
y_train, y_test = np.ravel(y_train), np.ravel(y_test)
```

Feel free to use our lovely NumPy `.shape()` methods to check the newly created splitted dataset sizes to ensure we're working with the appropriate data. 

Now that we have our training and test data, let's instantiate our k-Nearest Neighbors Classifier model. 

An important argument we can and should specify to our kNN classifier is the number of neighbors examined for every data point. 

kNN looks at every data point at least once across our entire dataset, but it's up to us to say for each data point, how many *k* neighbors it cross-references. 

---

Those of you who've dabbled with the constraints of Python programming may immediately be thinking that for larger k-values, this algorithm may become exponentially ineffective due to having to iterate over enormous quantities of redundant data.

You would be correct.

Therefore, it is imperative to use values for k that make relative sense with the scale of our data. Values over 1000 may not carry as much bearing on our data as values between 10 and 25, for instance. 

In our example, we're just going to use three (3) neighbors. 

```
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
```

Now that our model is instantiated and fitted to our training data, let's dive right in with making some predictions. 

This should be second-nature by now!

```
y_pred = knn.predict(X_test)
print(y_pred)
```

If we run `y_pred.shape` and `y_test.shape` in different cells, we can see that our resultant vector is of a shape consistent with what we expect. Great!

However, this doesn't tell us much without an explicit accuracy metric. Let's fix that, shall we? 

```
knn.score(X_test, y_test)
```

Hmm, not bad. 

**Can you think of any ways this model can be improved**? 

---

One last thing we can play with is seeing which accuracies work best for our model's fitness. 

As always, we can do this through some clever MatPlotLib and Python programming implementation.

```
neighbors = np.arange(1, 25)
train_accuracy, test_accuracy = list(), list()

for iterator, kterator in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=kterator)
    knn.fit(X_train, y_train)
    train_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))
    
plt.figure(figsize=[13, 8])
plt.plot(neighbors, test_accuracy, label="Testing Accuracy")
plt.plot(neighbors, train_accuracy, label="Training Accuracy")
plt.legend()
plt.title("Value VS Accuracy")
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.xticks(neighbors)
plt.savefig("knn_accuracies.png")
plt.show()

print("Best Accuracy is {} with K={}".format(np.max(test_accuracy), 1 + test_accuracy.index(np.max(test_accuracy))))
```

The visualization that you should see now adequately shows us our accuracy measurements across a range of *k* values from 1 to 25 for both our training and testing data segmentations.

In this case, we're trying to maximize our testing accuracy, so we want to grab the value of k where our blue line is at an absolute maximum.

**Can you think of other ways to tune our model to improve our score**? 

