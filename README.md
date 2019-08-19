# Dogs vs. Cats! Build Deep Neural Networks from scratch!

![](https://miro.medium.com/max/1500/1*36MELEhgZsPFuzlZvObnxA.gif)

## Introduction
This project will walk you through the process of building Deep Neural Networks (DNN) algorithms to predict dogs and cats from scratch (without using 3rd party libraries). The accuracy of this DNN will be compared to the one from SKLearn using Logistic Regression.


## Instalation
Please unzip data.zip file into data folder before running this jupyter notebook file.
Download data.zip: https://drive.google.com/file/d/1Mkqf8r_8mQLjUEsvkydGIoDxPIsXbb3Z/view?usp=sharing

----
### _Hands-on tutorial_: 

# Building a simple deep neural networks to classify dogs ands cat from scratch
----

### About the datasets

* The datasets contains train dataset (5000 images) and test dataset (500 images)  - which are in __H5__ format. __H5__ is an efficience way to store data, and Python has a library __h5py__ for working with __H5__.
* The train and test dataset contain 2 keys: `images` (which contains training instances) and `labels` (which has 2 values: 0 represents dogs and 1 represents cats).

![](https://scontent.fhan2-3.fna.fbcdn.net/v/t1.0-9/68587621_2334797396836942_7118842923717754880_n.jpg?_nc_cat=108&_nc_oc=AQlgOJJ7spaaYN4PQTE-QZBZ4fTKa6k7TIMYqErY-XoLNI313GAsv9oKP77GF_WekcI&_nc_ht=scontent.fhan2-3.fna&oh=e0fdbe54f1709564296a298d99c938c1&oe=5E118AC6)

----

### For the first part, we are going to build a simple neural network with 2 hidden layer
_If you have no idea what is neural network, [this article](https://www.techradar.com/news/what-is-a-neural-network) mights help_

![](https://scontent.fhan2-3.fna.fbcdn.net/v/t1.0-9/68657008_2334797366836945_8297018333513908224_n.jpg?_nc_cat=108&_nc_oc=AQmmlzAvMMimAaYhVmkDNYc0a73Da0cvQWlgbecDTRljZ-KuTbOQLT-qAAuNDztuXjs&_nc_ht=scontent.fhan2-3.fna&oh=0d5faf632840be9cf711d46cafb1e7f5&oe=5E12A59B)

### meaning of these symbols:
* __X__: Our input layer, in this case is our `images` of dog and cat
* __Yh__, __Y__: Predicted label (which is produced by neural network) and Real label
* __W1__, __W2__, __W3__: Our need-to-optimized parameters - these are bones of neural network.
* __A__, __Z__: These are informations that transfer through the network from input data (adjustments by multiply with W) and help us to produce the prediction - which then used for optimized W.
----

## The first step: Initialize the network
We are going to initialize W and b (bias) regarded to input layer and each hidden layer.
* _The input layer has 49,152 node, while each hidden layer has 3. The output layer contain 1 node that represent the label (0 for dog and 1 for cat)_ => the shape of W1, W2 and W3 are (3, 49.152), (3,3), (1,3) respectively.

```
W1 = np.random.randn(num_hidden, X_train.shape[1])/np.sqrt(X_train.shape[1])
b1 = np.random.randn(num_hidden, 1)

W2 = np.random.randn(num_hidden, num_hidden)/np.sqrt(num_hidden)
b2 = np.random.randn(num_hidden, 1)

W3 = np.random.randn(1, num_hidden)/np.sqrt(num_hidden)
b3 = np.random.randn(1, 1)
```

#### After initialize the network, we start out 1st step: feed the training data throught the network! 
----

## Second step: Feed data forward through layer.

* Remember that the shape of input and W must match, so we should check the shape before making any calculation.

![](https://scontent.fsgn2-2.fna.fbcdn.net/v/t1.0-9/68755490_2334797376836944_5889943932975972352_o.jpg?_nc_cat=102&_nc_oc=AQmdumJRVIeyxUm1_iraazG0lIBxV0NTC46QMWCkZKfQU-JYH4l-oQIwmqbxRwCjlNE&_nc_ht=scontent.fsgn2-2.fna&oh=f0154d2f35fa970746ff27ce8a80e7d0&oe=5DC822F5)

__Feed forward__

```
Z1 = np.dot(W1, X_train.T) + b1
A1 = Relu(Z1) 

Z2 = np.dot(W2, A1) + b2
A2 = Relu(Z2)

Z3 = np.dot(W3, A2) + b3
y_hat = Sigmoid(Z3)
```

__Calculate the cost function: J = $1/m*(-(y*\log(yhat) -(1-y)*\log(1-yhat))$__
>J = (- np.multiply(y_train, np.log(y_hat)) - np.multiply(1-y_train, np.log(1-y_hat))) / y_train.shape[1] 

## Third step: Back propagation
After having the cost J (Which present how big the different between predicts made by neural network and real labels), we compute derivative of J with respect to W1, W2 & W3 - which are used for adjusting W and b.
* Note that the formular for back-propagation at all layer are lookalike except for last layer.
* At each step we should check the shape of all factors to make sure that we are on the right track.

__Compute derivative__

```
e3 = Yh - Y
dW3 = e3.dot(A2.T)/A2.shape[0]
db3 = np.sum(e3)/A2.shape[0] 

e2 = W3.T.dot(e3) * A2
dW2 = e2.dot(A1.T)/A1.shape[0]
db2 = np.sum(e2)/A1.shape[0]```

e1 =  W2.T.dot(e2) * A1
dW1 = e1.dot(X_train)/X_train.shape[1]
db1 = np.sum(e1)/X_train.shape[1]
```

__Then we update W and b__

```lr = 0.001

W3 -= lr * dW3
W2 -= lr * dW2
W1 -= lr * dW1

b3 -= lr * db3
b2 -= lr * db2
b1 -= lr * db1
```

_`lr` is the "learning rate". The learning rate define how much we want our model learned at each iteration. If the learning rate is too big it will make the neural network never converge while the very small learning rate will take our model forever to converge._

### Congrat! We have just finished 1 round of feed-forward and backpropagation. Our parameters is making a small step toward the optimal paremeters and that is good. Normally we would run couple of thousands time in order to have the good-enough model that can be able to distinquish between dog and cat.

=> So it's time to put all that we have done in a for loop

```
lr = 0.01
for i in range(3000):
    num_train = np.random.randint(0, 12289)

    Z1 = np.dot(W1, X_train.T) + b1
    A1 = Relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = Relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    y_hat = Sigmoid(Z3)
    J = (- np.multiply(y_train, np.log(y_hat)) - np.multiply(1-y_train, np.log(1-y_hat))) / y_train.shape[1]

    e3 = y_hat - y_train
    dW3 = e3.dot(A2.T)/A2.shape[0]
    db3 = np.sum(e3)/A2.shape[0]
    e2 = W3.T.dot(e3) * A2
    dW2 = e2.dot(A1.T)/A1.shape[0]
    db2 = np.sum(e2)/A1.shape[0]
    e1 =  W2.T.dot(e2) * A1
    dW1 = e1.dot(X_train)/X_train.shape[1]
    db1 = np.sum(e1)/X_train.shape[1]

    W3 -= lr * dW3
    W2 -= lr * dW2
    W1 -= lr * dW1
    b3 -= lr * db3
    b2 -= lr * db2
    b1 -= lr * db1
```

## Further work:
- We should try to clean our code by putting them in `def` function.
- For now, we are using the parameter whose name is manually create => We should try to create a more flexible way for our parameter in case we want to increase (decrease) the number of hidden layer.
- If the number of traning is big, then the training time will increase very much. We could deal with that by using mini-batch. You can read about it [here](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/)