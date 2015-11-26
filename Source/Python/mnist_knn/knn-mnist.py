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

    # extracted_test_sample = extractFeatures(test_sample, [74,745,361,445,164,681,258,230,277,719,509,637,738,709,529,557,473,175,664,133,305,75,333,585,501,222,105,612,342,250,286,746,202,481,68,621,94,638,257,314,165,639,278,147,229,692,120,453,720,285,680,650,76,737,708,174,564,194,613,134,536,592,425,640,747,201,67,313,135,397,558,256,665,284,693,106,93,530,77,649,508,736,721,146,341,586,369,136,306,251,173,228,707,666,502,667,620,679,641,748,119,279,223,166,107,66,92,772,771,312,668,480,735,474,694,78,145,614,770,137,200,722,446,678,283,749,648,255,773,418,42,340,195,108,227,311,334,362,41,669,307,563,774,390,40,706,591,65,91,769,768,43,452,750,695,44,535,723,172,677,775,642,79,619,396,167,339,118,39,507,751,254,503,559,531,199,696,335,734,587,766,767,64,45,109,776,479,282,226,80,47,11,13,16,8,12,14,9,15,10,116,81,7,90,88,6,5,4,3,2,89,87,114,18,115,86,117,82,83,84,85,17,111,19,51,50,30,52,53,54,55,56,49,48,46,36,38,37,35,32,34,33,31,29,20,62,61,28,63,110,112,113,21,60,59,58,25,27,26,24,57,23,22,784,392,138,676,697,675,699,674,698,700,139,704,705,703,701,702,673,672,671,615,616,590,670,589,617,618,643,644,647,646,645,724,725,726,763,764,762,759,761,765,777,778,779,782,781,780,760,758,727,730,731,729,757,728,732,733,752,753,756,755,754,588,562,561,281,308,280,225,253,309,310,336,337,364,363,338,252,224,366,142,143,141,198,140,144,168,169,170,197,196,171,365,367,560,476,477,475,449,451,478,504,505,506,534,533,532,450,448,368,393,394,783,447,391,395,419,420,421,424,423,422,1])
    extracted_test_sample = test_sample

    preds = model.predict(extracted_test_sample)

    accuracy = model.score(extracted_test_sample, test_labels_sample)

    cm = metrics.confusion_matrix(test_labels_sample, preds)

    print("Confusion matrix:\n%s" % cm)

    plot_confusion_matrix(cm)

    digitsDict = digitDict.initData()
    print 'true \t\t predicted'
    errors = 0
    for i in xrange(0, len(test_sample)):
        if preds[i] != test_labels_sample[i]:
            errors += 1
            trueClass = str(int(test_labels_sample[i]))
            prediction = str(int(preds[i]))
            digitDict.updateErrors(digitsDict, trueClass, prediction)
            print test_labels_sample[i], '\t\t', preds[i]
    digitDict.save(digitsDict)

    print 'Accuracy: ', accuracy, '\nErrors: ', errors

def extractFeatures(sample, features):
    extrated_sample = np.empty([len(sample), 400])
    for i in range(len(sample)):
        extrated_sample[i] = np.delete(sample[i], features)

    return extrated_sample

def main():
    ## Read MNIST Data
    mnist = readData()

    train = mnist.data[:60000]
    # extracted_train = extractFeatures(train, [74,745,361,445,164,681,258,230,277,719,509,637,738,709,529,557,473,175,664,133,305,75,333,585,501,222,105,612,342,250,286,746,202,481,68,621,94,638,257,314,165,639,278,147,229,692,120,453,720,285,680,650,76,737,708,174,564,194,613,134,536,592,425,640,747,201,67,313,135,397,558,256,665,284,693,106,93,530,77,649,508,736,721,146,341,586,369,136,306,251,173,228,707,666,502,667,620,679,641,748,119,279,223,166,107,66,92,772,771,312,668,480,735,474,694,78,145,614,770,137,200,722,446,678,283,749,648,255,773,418,42,340,195,108,227,311,334,362,41,669,307,563,774,390,40,706,591,65,91,769,768,43,452,750,695,44,535,723,172,677,775,642,79,619,396,167,339,118,39,507,751,254,503,559,531,199,696,335,734,587,766,767,64,45,109,776,479,282,226,80,47,11,13,16,8,12,14,9,15,10,116,81,7,90,88,6,5,4,3,2,89,87,114,18,115,86,117,82,83,84,85,17,111,19,51,50,30,52,53,54,55,56,49,48,46,36,38,37,35,32,34,33,31,29,20,62,61,28,63,110,112,113,21,60,59,58,25,27,26,24,57,23,22,784,392,138,676,697,675,699,674,698,700,139,704,705,703,701,702,673,672,671,615,616,590,670,589,617,618,643,644,647,646,645,724,725,726,763,764,762,759,761,765,777,778,779,782,781,780,760,758,727,730,731,729,757,728,732,733,752,753,756,755,754,588,562,561,281,308,280,225,253,309,310,336,337,364,363,338,252,224,366,142,143,141,198,140,144,168,169,170,197,196,171,365,367,560,476,477,475,449,451,478,504,505,506,534,533,532,450,448,368,393,394,783,447,391,395,419,420,421,424,423,422,1])
    extracted_train = train

    test = mnist.data[60000:]

    train_labels = mnist.target[:60000]
    test_labels = mnist.target[60000:]

    model = KNeighborsClassifier(n_neighbors=3, algorithm='brute').fit(extracted_train, train_labels)

    ## Uncomment this line for a vizualization of errors
    # testDatasetWithVisualization(model, train, test, test_labels)

    testDataset(model, test, test_labels, 1)

main()