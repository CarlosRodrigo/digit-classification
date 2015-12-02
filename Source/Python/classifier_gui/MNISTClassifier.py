from sklearn.datasets import fetch_mldata
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pylab

def readData():
    DATA_PATH = '~/data'
    mnist = fetch_mldata('MNIST original', data_home=DATA_PATH)
    return mnist

def initClassifier():
    mnist = readData()

    train = mnist.data[:60000]
    test = mnist.data[60000:]

    train_labels = mnist.target[:60000]
    test_labels = mnist.target[60000:]

    model = KNeighborsClassifier(n_neighbors=3, algorithm='brute').fit(train, train_labels)

    return model

def predict(model, value):
    value = np.array(value)

    pylab.imshow(value.reshape(28, 28), cmap="Greys")
    pylab.show() 

    preds = model.predict(value)

    return preds