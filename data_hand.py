import random
import numpy as np
from Fei_dataset import *
from six.moves import xrange
from scipy.misc import imsave as ims
from HSICSupport import *
from ops import *
from Utlis2 import *
import gzip
import cv2
import keras as keras
#from copy import deepcopy
#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras import datasets

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def load_mnist(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec


def GiveMNIST_SVHN():
    mnistName = "mnist"
    data_X, data_y = load_mnist(mnistName)

    #data_X = np.expand_dims(data_X, axis=3)
    data_X = np.concatenate((data_X, data_X, data_X), axis=-1)

    size = (int(32), int(32))
    myArr = []
    for i in range(np.shape(data_X)[0]):
        image = cv2.resize(data_X[i],size, interpolation=cv2.INTER_AREA)
        myArr.append(image)

    data_X = np.array(myArr)

    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]
    y_train = data_y[0:60000]
    y_test = data_y[60000:70000]

    mnist_train_x = x_train
    mnist_train_label = y_train
    mnist_test = x_test
    mnist_label_test = y_test


    myTest = mnist_train_x[0:64]

    ims("results/" + "gggg" + str(0) + ".jpg", merge2(myTest[:64], [8, 8]))

    x_train, y_train, x_test, y_test = GetSVHN_DataSet()
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    return mnist_train_x,mnist_train_label,mnist_test,mnist_label_test,x_train,y_train,x_test,y_test

def GiveFashion_32():
    mnistName = "Fashion"
    data_X, data_y = load_mnist(mnistName)

    #data_X = np.expand_dims(data_X, axis=3)
    data_X = np.concatenate((data_X, data_X, data_X), axis=-1)

    size = (int(32), int(32))
    myArr = []
    for i in range(np.shape(data_X)[0]):
        image = cv2.resize(data_X[i],size, interpolation=cv2.INTER_AREA)
        myArr.append(image)

    data_X = np.array(myArr)

    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]
    y_train = data_y[0:60000]
    y_test = data_y[60000:70000]

    fashion_train_x = x_train
    fashion_train_label = y_train
    fashion_test = x_test
    fashion_label_test = y_test

    return fashion_train_x,fashion_train_label,fashion_test,fashion_label_test

def GiveFashion32():
    mnistName = "Fashion"
    data_X, data_y = load_mnist(mnistName)

    # data_X = np.expand_dims(data_X, axis=3)
    data_X = np.concatenate((data_X, data_X, data_X), axis=-1)

    size = (int(32), int(32))
    myArr = []
    for i in range(np.shape(data_X)[0]):
        image = cv2.resize(data_X[i], size, interpolation=cv2.INTER_AREA)
        myArr.append(image)

    data_X = np.array(myArr)

    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]
    y_train = data_y[0:60000]
    y_test = data_y[60000:70000]

    mnist_train_x = x_train
    mnist_train_label = y_train
    mnist_test = x_test
    mnist_label_test = y_test

    return mnist_train_x,mnist_train_label,mnist_test,mnist_label_test

def ReturnSet_ByIndex(x,y,startIndex,endIndex):
    xarr = []
    yarr = []
    difference = endIndex - 10
    for i in range(np.shape(x)[0]):
        if y[i] >= startIndex and y[i] <= endIndex:
            xarr.append(x[i])
            label = y[i] - difference
            label = label-1
            yarr.append(label)

    xarr = np.array(xarr)
    yarr = np.array(yarr)
    return xarr,yarr

def Split_CIFAR100_ReturnTesting():
    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

    x_train = x_train/255
    x_test = x_test/ 255

    from keras.utils.np_utils import to_categorical

    x1_,y1_ = ReturnSet_ByIndex(x_test,y_test,1,10)
    x2_,y2_ = ReturnSet_ByIndex(x_test,y_test,11,20)
    x3_,y3_ = ReturnSet_ByIndex(x_test,y_test,21,30)
    x4_,y4_ = ReturnSet_ByIndex(x_test,y_test,31,40)
    x5_,y5_ = ReturnSet_ByIndex(x_test,y_test,41,50)

    y1_ = to_categorical(y1_, num_classes=None)
    y2_ = to_categorical(y2_, num_classes=None)
    y3_ = to_categorical(y3_, num_classes=None)
    y4_ = to_categorical(y4_, num_classes=None)
    y5_ = to_categorical(y5_, num_classes=None)

    return x1_,y1_,x2_,y2_,x3_,y3_,x4_,y4_,x5_,y5_

def Split_CIFAR100():
    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

    x_train = x_train/255
    x_test = x_test/ 255
    x1,y1 = ReturnSet_ByIndex(x_train,y_train,1,10)
    x2,y2 = ReturnSet_ByIndex(x_train,y_train,11,20)
    x3,y3 = ReturnSet_ByIndex(x_train,y_train,21,30)
    x4,y4 = ReturnSet_ByIndex(x_train,y_train,31,40)
    x5,y5 = ReturnSet_ByIndex(x_train,y_train,41,50)

    x1_,y1_ = ReturnSet_ByIndex(x_test,y_test,1,10)
    x2_,y2_ = ReturnSet_ByIndex(x_test,y_test,11,20)
    x3_,y3_ = ReturnSet_ByIndex(x_test,y_test,21,30)
    x4_,y4_ = ReturnSet_ByIndex(x_test,y_test,31,40)
    x5_,y5_ = ReturnSet_ByIndex(x_test,y_test,41,50)

    x_ = np.concatenate((x1_,x2_,x3_,x4_,x5_),axis=0)
    y_ = np.concatenate((y1_,y2_,y3_,y4_,y5_),axis=0)

    return x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x_,y_

Split_CIFAR100()

def Split_dataset(x,y,n_label):
    y = np.argmax(y,axis=1)
    n_each = n_label / 10
    isRun = True
    x_train = []
    y_train = []
    index = np.zeros(10)
    while(isRun):
        a = random.randint(0, np.shape(x)[0])-1
        x1 = x[a]
        y1 = y[a]
        if index[y1] < n_each:
            x_train.append(x1)
            y_train.append(y1)
            index[y1] = index[y1]+1
        isOk1 = True
        for i in range(10):
            if index[i] < n_each:
                isOk1 = False
        if isOk1:
            break

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train,y_train

def Give_InverseFashion32():
    mnistName = "Fashion"
    data_X, data_y = load_mnist(mnistName)
    data_X = np.reshape(data_X, (-1, 28, 28))
    for i in range(np.shape(data_X)[0]):
        for k1 in range(28):
            for k2 in range(28):
                data_X[i, k1, k2] = 1.0 - data_X[i, k1, k2]

    data_X = np.reshape(data_X, (-1, 28, 28, 1))
    data_X = np.concatenate((data_X, data_X, data_X), axis=-1)
    size = (int(32), int(32))
    myArr = []
    for i in range(np.shape(data_X)[0]):
        image = cv2.resize(data_X[i], size, interpolation=cv2.INTER_AREA)
        myArr.append(image)

    data_X = np.array(myArr)

    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]
    y_train = data_y[0:60000]
    y_test = data_y[60000:70000]

    mnist_train_x = x_train
    mnist_train_label = y_train
    mnist_test = x_test
    mnist_label_test = y_test

    return mnist_train_x,mnist_train_label,mnist_test,mnist_label_test

def GiveMNIST32():
    mnistName = "mnist"
    data_X, data_y = load_mnist(mnistName)

    # data_X = np.expand_dims(data_X, axis=3)
    data_X = np.concatenate((data_X, data_X, data_X), axis=-1)

    size = (int(32), int(32))
    myArr = []
    for i in range(np.shape(data_X)[0]):
        image = cv2.resize(data_X[i], size, interpolation=cv2.INTER_AREA)
        myArr.append(image)

    data_X = np.array(myArr)

    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]
    y_train = data_y[0:60000]
    y_test = data_y[60000:70000]

    mnist_train_x = x_train
    mnist_train_label = y_train
    mnist_test = x_test
    mnist_label_test = y_test

    return mnist_train_x, mnist_train_label, mnist_test, mnist_label_test
