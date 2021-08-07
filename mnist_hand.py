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

def Split_dataset_by5(x,y):
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    arr5 = []
    labelArr1 = []
    labelArr2 = []
    labelArr3 = []
    labelArr4 = []
    labelArr5 = []

    n = np.shape(x)[0]
    for i in range(n):
        data1 = x[i]
        label1 = y[i]
        if label1[0] == 1 or label1[1] == 1:
            arr1.append(data1)
            labelArr1.append(label1)

        if label1[2] == 1 or label1[3] == 1:
            arr2.append(data1)
            labelArr2.append(label1)

        if label1[4] == 1 or label1[5] == 1:
            arr3.append(data1)
            labelArr3.append(label1)

        if label1[6] == 1 or label1[7] == 1:
            arr4.append(data1)
            labelArr4.append(label1)

        if label1[8] == 1 or label1[9] == 1:
            arr5.append(data1)
            labelArr5.append(label1)

    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    arr3 = np.array(arr3)
    arr4 = np.array(arr4)
    arr5 = np.array(arr5)

    labelArr1 = np.array(labelArr1)
    labelArr2 = np.array(labelArr2)
    labelArr3 = np.array(labelArr3)
    labelArr4 = np.array(labelArr4)
    labelArr5 = np.array(labelArr5)
    return arr1,labelArr1,arr2,labelArr2,arr3,labelArr3,arr4,labelArr4,arr5,labelArr5

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

def Give_InverseFashion():
    mnistName = "Fashion"
    data_X, data_y = load_mnist(mnistName)
    data_X = np.reshape(data_X,(-1,28,28))
    for i in range(np.shape(data_X)[0]):
        for k1 in range(28):
            for k2 in range(28):
                data_X[i,k1,k2] = 1.0 - data_X[i,k1,k2]

    data_X = np.reshape(data_X,(-1,28,28,1))
    return data_X,data_y

def Give_InverseMNIST32():
    mnistName = "mnist"
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
