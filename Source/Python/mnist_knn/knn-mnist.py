import pylab
import matplotlib.pyplot as plt
import numpy as np
import digitDict
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import metrics

'''
    Function that reads MNIST Data
'''  
def readData():
    DATA_PATH = '~/data'
    mnist = fetch_mldata('MNIST original', data_home=DATA_PATH)
    return mnist

'''
    Finding out data dimensions
'''
def shapeInformation(mnist):
    print mnist.data.shape

    row = mnist.data[0,:] # First row of the array
    col = mnist.data[:,0] # First column of the array
    
    print row.shape
    print col.shape

'''
    function that draws a visualization of one digit
    from the MNIST dataset
'''
def visualizeData(mnist, train, digit):
    img = mnist.data[digit]

    pylab.imshow(img.reshape(28, 28), cmap="Greys")
    pylab.imshow(train[digit].reshape(28, 28), cmap="Greys")
    pylab.show()

'''
    Plots images a given imagem from the MNIST dataset and its neighbors
'''
def plot_digits_neighbors(img, imgs, n):
    fig = pylab.figure()
    fig.add_subplot(3, 3, 2, xticklabels=[], yticklabels=[])
    pylab.imshow(img.reshape(28, 28), cmap="Greys")
    for i in xrange(0, n):
        fig.add_subplot(1, n, i, xticklabels=[], yticklabels=[])
        img = imgs[i]
        pylab.imshow(img.reshape(28, 28), cmap="Greys")

'''
    Plots a confusion matrix
'''
def plot_cm(cm):
    pylab.matshow(np.log(1+cm))
    pylab.show()

'''
    Plots a confusion matrix with the number of misclassifications
    on each position of the matrix
'''
def plot_confusion_matrix(cm):
    norm_conf = []
    for i in cm:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')

    width = len(cm)
    height = len(cm[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = '0123456789'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.show()

def testDatasetWithVisualization(model, train, test, test_labels, n_slice=50):
    test_sample = test[::n_slice]
    test_labels_sample = test_labels[::n_slice]

    preds = model.predict(test_sample)

    accuracy =  model.score(test_sample, test_labels_sample)

    for i in xrange(0, len(test_sample)):
        if preds[i] != test_labels_sample[i]:
            print test_labels_sample[i], preds[i]
            query_img = test_sample[i]
            distance, result = model.kneighbors(query_img, n_neighbors=5)
            plot_digits_neighbors(query_img, train[result[0],:], len(result[0]))
            pylab.show()

    print 'Accuracy: ', accuracy

def testDataset(model, test, test_labels, n_slice=1):
    test_sample = test[::n_slice]
    test_labels_sample = test_labels[::n_slice]
    # print test_sample[1]

    preds = model.predict(test_sample)

    accuracy = model.score(test_sample, test_labels_sample)

    cm = metrics.confusion_matrix(test_labels_sample, preds)

    print("Confusion matrix:\n%s" % cm)

    plot_confusion_matrix(cm)

    digitsDict = digitDict.initData()
    print 'true \t\t predicted'
    for i in xrange(0, len(test_sample)):
        if preds[i] != test_labels_sample[i]:
            trueClass = str(int(test_labels_sample[i]))
            prediction = str(int(preds[i]))
            digitDict.updateErrors(digitsDict, trueClass, prediction)
            print test_labels_sample[i], '\t\t', preds[i]
    digitDict.save(digitsDict)

    print 'Accuracy: ', accuracy

def main():
    ## Read MNIST Data
    mnist = readData()

    train = mnist.data[:60000]

    test = mnist.data[60000:]

    train_labels = mnist.target[:60000]
    test_labels = mnist.target[60000:]

    model = KNeighborsClassifier(n_neighbors=3, algorithm='brute').fit(train, train_labels)

    ## Uncomment this line for a vizualization of errors
    # testDatasetWithVisualization(model, train, test, test_labels)

    testDataset(model, test, test_labels, 5)

main()