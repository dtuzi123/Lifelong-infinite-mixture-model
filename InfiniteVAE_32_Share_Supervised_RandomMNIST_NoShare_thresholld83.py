import time
from utils import *
from scipy.misc import imsave as ims
from ops import *
from utils import *
from Utlis2 import *
import random as random
from glob import glob
import os, gzip
from glob import glob
from Basic_structure import *
from mnist_hand import *
from CIFAR10 import *
from MMD_Lib import *
from data_hand import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def file_name(file_dir):
    t1 = []
    file_dir = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/"
    for root, dirs, files in os.walk(file_dir):
        for a1 in dirs:
            b1 = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/" + a1 + "/renders/*.png"
            b1 = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/" + a1
            for root2, dirs2, files2 in os.walk(b1):
                for c1 in dirs2:
                    b2 = b1 + "/" + c1 + "/*.png"
                    img_path = glob(b2)
                    t1.append(img_path)
    cc = []

    for i in range(len(t1)):
        a1 = t1[i]
        for p1 in a1:
            cc.append(p1)
    return cc

def Image_classifier(inputs, scopename, is_training=True, reuse=False):
    with tf.variable_scope(scopename, reuse=reuse):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            x = tf.reshape(inputs, [-1, 32, 32, 3])

            # For slim.conv2d, default argument values are like
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # padding='SAME', activation_fn=nn.relu,
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            net = slim.conv2d(x, 32, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.flatten(net, scope='flatten3')

            # For slim.fully_connected, default argument values are like
            # activation_fn = nn.relu,
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            net = slim.fully_connected(net, 1024, scope='fc3')
            net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default
            outputs = slim.fully_connected(net, 4, activation_fn=None, normalizer_fn=None, scope='fco')
    return outputs

def CodeImage_classifier(s, scopename, reuse=False):
    with tf.variable_scope(scopename, reuse=reuse):
        input = s

        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)
        n_hidden = 500
        keep_prob = 0.9

        # 1st hidden layer
        w0 = tf.get_variable('w0', [input.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(s, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        n_output = 4
        # output layer-mean
        wo = tf.get_variable('wo', [h0.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y1 = tf.matmul(h0, wo) + bo
        y = tf.nn.softmax(y1)

    return y1, y

def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def my_gumbel_softmax_sample(logits, cats_range, temperature=0.1):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    logits_with_noise = tf.nn.softmax(y / temperature)
    return logits_with_noise

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

def My_Encoder_mnist(image, z_dim, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        len_discrete_code = 4

        is_training = True
        x = image
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='c_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='c_conv2'), is_training=is_training, scope='c_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

        net = lrelu(bn(linear(net, 64, scope='e_fc11'), is_training=is_training, scope='c_bn11'))

        z_mean = linear(net, z_dim, 'e_mean')
        z_log_sigma_sq = linear(net, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq

def My_Classifier_mnist(image, z_dim, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        len_discrete_code = 4

        is_training = True
        # z_dim = 32
        x = image
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='c_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='c_conv2'), is_training=is_training, scope='c_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

        net = lrelu(bn(linear(net, 64, scope='e_fc11'), is_training=is_training, scope='c_bn11'))

        out_logit = linear(net, len_discrete_code, scope='e_fc22')
        softmaxValue = tf.nn.softmax(out_logit)

        return out_logit, softmaxValue

def MINI_Classifier(s, scopename, reuse=False):
    keep_prob = 1.0
    with tf.variable_scope(scopename, reuse=reuse):
        input = s
        n_output = 10
        n_hidden = 500
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [input.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(s, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        n_output = 10
        # output layer-mean
        wo = tf.get_variable('wo', [h0.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y1 = tf.matmul(h0, wo) + bo
        y = tf.nn.softmax(y1)

    return y1, y

# Create model of CNN with slim api

class LifeLone_MNIST(object):
    def __init__(self):
        self.batch_size = 64
        self.input_height = 28
        self.input_width = 28
        self.c_dim = 1
        self.z_dim = 100
        self.len_discrete_code = 4
        self.epoch = 100

        self.componentCount = 1
        self.currentComponent = 1
        self.fid_hold = 150
        self.IsAdd = 0

        self.learning_rate = 0.0002
        self.beta1 = 0.5

        self.VAE_List = []
        self.VAE_GeneratorList = []
        self.EncoderMean_List = []
        self.EncoderVariance_List = []
        self.Feature_List = []
        self.Elbo_List = []
        self.classifierPredictionList = []

        # MNIST dataset
        mnistName = "mnist"
        fashionMnistName = "Fashion"

        datax, datay = load_mnist("mnist")
        self.x_train = datax[0:60000]
        self.x_test = datax[60000:70000]
        self.y_train = datay[0:60000]
        self.y_test = datay[60000:70000]

        num_tasks_to_run = 10
        self.task_permutation = []
        for task in range(num_tasks_to_run):
            self.task_permutation.append(np.random.permutation(28*28))

    def Create_NewExpert(self):

        encoder_myStr = "z_encoder" + str(self.componentCount)
        myStr = "encoder" + str(self.componentCount)
        myStr2 = "VAE_Generator" + str(self.componentCount)

        # encoder continoual information
        a1 = Generator_SharedMNIST("VAE_Generator_Shared", tf.concat((self.z,self.labels),axis=1), reuse=True)
        G1 = Generator_SubMNIST(myStr2, a1, reuse=False)
        self.VAE_GeneratorList.append(G1)

        # encoder continoual information
        logit,softmaxValue = Encoder_MNIST_Supervised_Small(self.inputs,self.z, myStr, batch_size=64, reuse=False)
        softmaxValue1 = tf.nn.softmax(logit)
        log_y1 = tf.log(softmaxValue1 + 1e-10)
        discrete1 = my_gumbel_softmax_sample(log_y1, np.arange(10))

        label_softmax = tf.nn.softmax(logit)
        predictions = tf.argmax(label_softmax, 1)
        self.classifierPredictionList.append(predictions)

        self.Shared_Encoder_z = Encoder_SVHN_Shared(self.inputs, "z_encoder_shared", batch_size=64, reuse=True)
        z_mean1, z_log_sigma_sq1, features = Encoder_SVHN_Specific(self.Shared_Encoder_z, encoder_myStr, batch_size=64,
                                                                   reuse=False)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)

        code = tf.concat((continous_variables1,discrete1),axis=1)
        a2 = Generator_SharedMNIST("VAE_Generator_Shared", code, reuse=True)

        VAE1 = Generator_SubMNIST(myStr2, a2, reuse=True)
        self.VAE_List.append(VAE1)

        classLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=self.labels))
        reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(VAE1 - self.inputs), [1, 2, 3]))

        KL_divergence1 = 0.5 * tf.reduce_sum(
            tf.square(z_mean1) + tf.square(z_log_sigma_sq1) - tf.log(1e-8 + tf.square(z_log_sigma_sq1)) - 1,
            1)
        KL_divergence1 = tf.reduce_mean(KL_divergence1)
        vaeloss1 = reconstruction_loss1 + KL_divergence1
        self.Elbo_List.append(vaeloss1)

    def GenerateBatachImages(self,batchImages):
        if self.componentCount == 1:
            return batchImages

        myArr = []
        myCount = int(self.batch_size/(self.componentCount+1))
        t1 = 0
        for i in range(self.componentCount):
            aa = i + 1
            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
            if aa != self.currentComponent:
                t1 = self.sess.run(self.VAE_GeneratorList[i],feed_dict={self.z:batch_z})

                for k in range(myCount):
                    myArr.append(t1[k])

        myCount = self.batch_size - np.shape(myArr)[0]
        for i in range(myCount):
            myArr.append(batchImages[i])
        myArr = np.array(myArr)
        return myArr

    def Activate_Expert(self, index):

        encoder_myStr = "z_encoder" + str(index)
        myStr = "encoder" + str(index)
        myStr2 = "VAE_Generator" + str(index)

        # encoder continoual information
        logit, softmaxValue = Encoder_MNIST_Supervised_Small(self.inputs,self.z, myStr, batch_size=64, reuse=True)
        softmaxValue1 = tf.nn.softmax(logit)
        log_y1 = tf.log(softmaxValue1 + 1e-10)
        discrete1 = my_gumbel_softmax_sample(log_y1, np.arange(10))

        self.Shared_Encoder_z = Encoder_SVHN_Shared(self.inputs, "z_encoder_shared", batch_size=64, reuse=True)
        z_mean1, z_log_sigma_sq1, features = Encoder_SVHN_Specific(self.Shared_Encoder_z, encoder_myStr, batch_size=64,
                                                                   reuse=True)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)

        code = tf.concat((continous_variables1,discrete1),axis=1)

        self.Shared_Decoder = Generator_SharedMNIST("VAE_Generator_Shared", code, reuse=True)

        VAE1 = Generator_SubMNIST(myStr2, self.Shared_Decoder, reuse=True)
        self.classLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=self.labels))
        reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(VAE1 - self.inputs), [1, 2, 3]))
        KL_divergence1 = 0.5 * tf.reduce_sum(
            tf.square(z_mean1) + tf.square(z_log_sigma_sq1) - tf.log(1e-8 + tf.square(z_log_sigma_sq1)) - 1,
            1)
        KL_divergence1 = tf.reduce_mean(KL_divergence1)

        ELBO = reconstruction_loss1+KL_divergence1
        self.vaeLoss = ELBO

        T_vars = tf.trainable_variables()
        GAN_generator_vars1 = [var for var in T_vars if var.name.startswith(myStr2)]
        Encoder_vars1 = [var for var in T_vars if var.name.startswith(myStr)]
        Encoder_vars1_z = [var for var in T_vars if var.name.startswith(encoder_myStr)]
        vae_vars = GAN_generator_vars1+Encoder_vars1+Encoder_vars1_z
        with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
            self.vae_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.vaeLoss, var_list=vae_vars)
            self.classifier_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.classLoss, var_list=vae_vars)

        global_vars = tf.global_variables()
        is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        self.sess.run(tf.variables_initializer(not_initialized_vars))

    def Predictions_By_Index2(self,testX,index):
        myPredict = self.classifierPredictionList[index - 1]

        totalN = np.shape(testX)[0]
        myN = int(totalN / self.batch_size)
        myPrediction = myPredict
        totalPredictions = []
        myCount = 0
        for i in range(myN):
            my1 = testX[self.batch_size * i:self.batch_size * (i + 1)]
            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
            predictions = self.sess.run(myPrediction, feed_dict={self.inputs: my1,self.z:batch_z})
            for k in range(self.batch_size):
                totalPredictions.append(predictions[k])

        totalPredictions = np.array(totalPredictions)
        return totalPredictions

    def Calculate_Accuracy_ByIndex(self, testX,testY, index):
        p1 = int(np.shape(testX)[0] / self.batch_size)
        myPro = self.Predictions_By_Index2(testX,index)
        target = [np.argmax(one_hot)for one_hot in testY]
        sumError = 0
        accCount = 0
        for i in range(np.shape(myPro)[0]):
            isState = True

            if myPro[i] == target[i]:
                accCount = accCount + 1

        totalCount = np.shape(myPro)[0]
        acc = float(accCount/totalCount)

        return acc

    def build_model(self):
        min_value = 1e-10
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.len_discrete_code])
        self.labels = tf.placeholder(tf.float32, [self.batch_size, 10])
        self.weights = tf.placeholder(tf.float32, [self.batch_size, 4])
        self.index = tf.placeholder(tf.int32, [self.batch_size])
        self.gan_inputs = tf.placeholder(tf.float32, [bs] + image_dims)
        self.gan_domain = tf.placeholder(tf.float32, [self.batch_size, 4])
        self.gan_domain_labels = tf.placeholder(tf.float32, [self.batch_size, 1])

        z_dim = 150
        # encoder continoual information
        logit,softmaxValue = Encoder_MNIST_Supervised_Small(self.inputs,self.z, "encoder1", batch_size=64, reuse=False)
        softmaxValue1 = tf.nn.softmax(logit)
        log_y1 = tf.log(softmaxValue1 + 1e-10)
        discrete1 = my_gumbel_softmax_sample(log_y1, np.arange(10))

        label_softmax = tf.nn.softmax(logit)
        predictions = tf.argmax(label_softmax, 1)
        self.classifierPredictionList.append(predictions)

        self.Shared_Encoder_z = Encoder_SVHN_Shared(self.inputs, "z_encoder_shared", batch_size=64, reuse=False)
        z_mean1, z_log_sigma_sq1, features = Encoder_SVHN_Specific(self.Shared_Encoder_z, "z_encoder1", batch_size=64,
                                                                   reuse=False)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)

        code = tf.concat((continous_variables1,discrete1),axis=1)
        self.Shared_Decoder = Generator_SharedMNIST("VAE_Generator_Shared", code, reuse=False)

        VAE1 = Generator_SubMNIST("VAE_Generator1", self.Shared_Decoder, reuse=False)

        a1 = Generator_SharedMNIST("VAE_Generator_Shared", tf.concat((self.z,self.labels),axis=1), reuse=True)
        GAN1 = Generator_SubMNIST("VAE_Generator1", a1, reuse=True)
        self.VAE_GeneratorList.append(GAN1)
        self.VAE_List.append(VAE1)

        self.classLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=self.labels))

        reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(VAE1 - self.inputs), [1, 2, 3]))

        KL_divergence1 = 0.5 * tf.reduce_sum(
            tf.square(z_mean1) + tf.square(z_log_sigma_sq1) - tf.log(1e-8 + tf.square(z_log_sigma_sq1)) - 1,
            1)
        KL_divergence1 = tf.reduce_mean(KL_divergence1)

        vaeloss1 = reconstruction_loss1 + KL_divergence1
        self.vaeLoss = vaeloss1
        self.Elbo_List.append(vaeloss1)

        tf.add_to_collection("VAE_loss1",vaeloss1)
        tf.add_to_collection("VAE_Generator1",GAN1)
        tf.add_to_collection("VAE_Reco1",VAE1)

        # Get VAE loss

        """ Training """
        # divide trainable variables into a group for D and a group for G
        T_vars = tf.trainable_variables()
        VAE_encoder_vars = [var for var in T_vars if var.name.startswith('encoder1')]
        VAE_generator_vars = [var for var in T_vars if var.name.startswith('VAE_Generator1')]
        VAE_generator_Shared_vars = [var for var in T_vars if var.name.startswith('VAE_Generator_Shared')]
        VAE_encoder_Shared_vars_z = [var for var in T_vars if var.name.startswith('z_encoder1')]
        VAE_encoder_Shared_vars_z_1 = [var for var in T_vars if var.name.startswith('z_encoder_shared')]

        vae_vars = VAE_encoder_vars + VAE_generator_vars+VAE_generator_Shared_vars+VAE_encoder_Shared_vars_z+VAE_encoder_Shared_vars_z_1

        # optimizers
        with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
            self.vae_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.vaeLoss, var_list=vae_vars)
            self.classifier_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
            .minimize(self.classLoss, var_list=vae_vars)
        b1 = 0

    def predict(self):
        # define the classifier
        label_logits = Image_classifier(self.inputs, "label_classifier", reuse=True)
        label_softmax = tf.nn.softmax(label_logits)
        predictions = tf.argmax(label_softmax, 1, name="predictions")
        return predictions

    def DomainPredict(self):
        z_mean1, z_log_sigma_sq1 = Encoder_SVHN(self.inputs, "encoder", batch_size=64, reuse=True)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1,
                                                                            dtype=tf.float32)

        domain_logit, domain_class = CodeImage_classifier(continous_variables1, "encoder_domain", reuse=True)
        log_y = tf.log(tf.nn.softmax(domain_logit) + 1e-10)
        discrete_real = my_gumbel_softmax_sample(log_y, np.arange(self.len_discrete_code))

        label_softmax = tf.nn.softmax(discrete_real)
        predictions = tf.argmax(label_softmax, 1, name="domainPredictions")
        return predictions

    def Give_predictedLabels(self, testX):
        totalN = np.shape(testX)[0]
        myN = int(totalN / self.batch_size)
        myPrediction = self.predict()
        totalPredictions = []
        myCount = 0
        for i in range(myN):
            my1 = testX[self.batch_size * i:self.batch_size * (i + 1)]
            predictions = self.sess.run(myPrediction, feed_dict={self.inputs: my1})
            for k in range(self.batch_size):
                totalPredictions.append(predictions[k])

        totalPredictions = np.array(totalPredictions)
        totalPredictions = keras.utils.to_categorical(totalPredictions)
        return totalPredictions

    def Give_DomainpredictedLabels(self, testX):
        totalN = np.shape(testX)[0]
        myN = int(totalN / self.batch_size)
        myPrediction = self.DomainPredict()
        totalPredictions = []
        myCount = 0
        for i in range(myN):
            my1 = testX[self.batch_size * i:self.batch_size * (i + 1)]
            predictions = self.sess.run(myPrediction, feed_dict={self.inputs: my1})
            for k in range(self.batch_size):
                totalPredictions.append(predictions[k])

        totalPredictions = np.array(totalPredictions)
        totalPredictions = keras.utils.to_categorical(totalPredictions, 4)
        return totalPredictions

    def Calculate_DomainAcc(self, testX, testY):
        # testX = self.mnist_test_x
        totalN = np.shape(testX)[0]
        myN = int(totalN / self.batch_size)
        myPrediction = self.DomainPredict()
        totalPredictions = []
        myCount = 0
        for i in range(myN):
            my1 = testX[self.batch_size * i:self.batch_size * (i + 1)]
            predictions = self.sess.run(myPrediction, feed_dict={self.inputs: my1})
            for k in range(self.batch_size):
                totalPredictions.append(predictions[k])

        totalPredictions = np.array(totalPredictions)

        testLabels = testY[0:np.shape(totalPredictions)[0]]
        testLabels = np.argmax(testLabels, 1)
        trueCount = 0
        for k in range(np.shape(testLabels)[0]):
            if testLabels[k] == totalPredictions[k]:
                trueCount = trueCount + 1

        accuracy = (float)(trueCount / np.shape(testLabels)[0])

        return accuracy

    def Calculate_accuracy(self, testX, testY):
        # testX = self.mnist_test_x
        totalN = np.shape(testX)[0]
        myN = int(totalN / self.batch_size)
        myPrediction = self.predict()
        totalPredictions = []
        myCount = 0
        for i in range(myN):
            my1 = testX[self.batch_size * i:self.batch_size * (i + 1)]
            predictions = self.sess.run(myPrediction, feed_dict={self.inputs: my1})
            for k in range(self.batch_size):
                totalPredictions.append(predictions[k])

        totalPredictions = np.array(totalPredictions)

        testLabels = testY[0:np.shape(totalPredictions)[0]]
        testLabels = np.argmax(testLabels, 1)
        trueCount = 0
        for k in range(np.shape(testLabels)[0]):
            if testLabels[k] == totalPredictions[k]:
                trueCount = trueCount + 1

        accuracy = (float)(trueCount / np.shape(testLabels)[0])

        return accuracy

    def Give_RealReconstruction(self):
        z_mean1, z_log_sigma_sq1 = Encoder_SVHN(self.inputs, "encoder", batch_size=64, reuse=True)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)

        domain_logit, domain_class = CodeImage_classifier(continous_variables1, "encoder_domain", reuse=True)
        log_y = tf.log(tf.nn.softmax(domain_logit) + 1e-10)
        discrete_real = my_gumbel_softmax_sample(log_y, np.arange(self.len_discrete_code))

        code = tf.concat((continous_variables1, discrete_real), axis=1)
        VAE1 = Generator_SVHN("VAE_Generator", code, reuse=True)

        reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(VAE1 - self.inputs), [1, 2, 3]))

        return reconstruction_loss1

    def Give_Elbo(self):
        z_mean1, z_log_sigma_sq1 = Encoder_SVHN(self.inputs, "encoder", batch_size=64, reuse=True)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)

        domain_logit, domain_class = CodeImage_classifier(continous_variables1, "encoder_domain", reuse=True)
        log_y = tf.log(tf.nn.softmax(domain_logit) + 1e-10)
        discrete_real = my_gumbel_softmax_sample(log_y, np.arange(self.len_discrete_code))

        code = tf.concat((continous_variables1, discrete_real), axis=1)
        VAE1 = Generator_SVHN("VAE_Generator", code, reuse=True)

        reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(VAE1 - self.inputs), [1, 2, 3]))

        y_labels = tf.argmax(discrete_real, 1)
        y_labels = tf.cast(y_labels, dtype=tf.float32)
        y_labels = tf.reshape(y_labels, (-1, 1))

        KL_divergence1 = 0.5 * tf.reduce_sum(
            tf.square(z_mean1 - y_labels) + tf.square(z_log_sigma_sq1) - tf.log(1e-8 + tf.square(z_log_sigma_sq1)) - 1,
            1)
        KL_divergence1 = tf.reduce_mean(KL_divergence1)

        return reconstruction_loss1 + KL_divergence1

    def Give_Elbo_byVAE(self,index):
        str1 = "encoder"+str(index)
        str2 = "VAE_Generator"+str(index)

        self.Shared_Encoder = Encoder_SVHN_Shared(self.inputs, "encoder_shared", batch_size=64, reuse=True)

        z_mean1, z_log_sigma_sq1,_ = Encoder_SVHN_Specific(self.Shared_Encoder, str1, batch_size=64, reuse=True)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)

        code = continous_variables1

        self.Shared_Decoder = Generator_SharedMNIST("VAE_Generator_Shared", code, reuse=True)
        VAE1 = Generator_SubMNIST(str2, self.Shared_Decoder, reuse=True)

        reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(VAE1 - self.inputs), [1, 2, 3]))

        KL_divergence1 = 0.5 * tf.reduce_sum(
            tf.square(z_mean1) + tf.square(z_log_sigma_sq1) - tf.log(1e-8 + tf.square(z_log_sigma_sq1)) - 1,
            1)
        KL_divergence1 = tf.reduce_mean(KL_divergence1)

        return reconstruction_loss1 + KL_divergence1

    def Calculate_ReconstructionErrors(self, testX):
        p1 = int(np.shape(testX)[0] / self.batch_size)
        myPro = self.Give_RealReconstruction()
        sumError = 0
        for i in range(p1):
            g = testX[i * self.batch_size:(i + 1) * self.batch_size]
            sumError = sumError + self.sess.run(myPro, feed_dict={self.inputs: g})

        sumError = sumError / p1
        return sumError

    def Calculate_Elbo(self, testX):
        p1 = int(np.shape(testX)[0] / self.batch_size)
        myPro = self.Give_Elbo()
        sumError = 0
        for i in range(p1):
            g = testX[i * self.batch_size:(i + 1) * self.batch_size]
            sumError = sumError + self.sess.run(myPro, feed_dict={self.inputs: g})

        sumError = sumError / p1
        return sumError

    def Calculate_Elbo_ByIndex(self, testX,index):
        p1 = int(np.shape(testX)[0] / self.batch_size)
        myPro = self.Give_Elbo_byVAE(index)
        sumError = 0
        for i in range(p1):
            g = testX[i * self.batch_size:(i + 1) * self.batch_size]
            sumError = sumError + self.sess.run(myPro, feed_dict={self.inputs: g})

        sumError = sumError / p1
        return sumError

    def test(self):

        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config.gpu_options.allow_growth = True

        #self.Create_NewExpert_ByIndex(2)
        #self.Create_NewExpert_ByIndex(3)
        self.componentCount = 3

        #new_saver = tf.train.import_meta_graph('models/InfiniteVAE.meta')
        with tf.Session(config=config) as sess:
            #new_saver.restore(sess, 'models/InfiniteVAE_Share.ckpt')
            self.sess = sess

            index1 = self.Select_Expert_ByData(self.mnist_test_x)
            mnistError = self.Calculate_Elbo_ByIndex(self.mnist_test_x,index1)

            index2 = self.Select_Expert_ByData(self.FashionTest_x)
            fashionError = self.Calculate_Elbo_ByIndex(self.FashionTest_x, index2)

            index3 = self.Select_Expert_ByData(self.svhn_test_x)
            svhnError = self.Calculate_Elbo_ByIndex(self.svhn_test_x, index3)

            index4 = self.Select_Expert_ByData(self.InverseFashionTest_x)
            IFashionError = self.Calculate_Elbo_ByIndex(self.InverseFashionTest_x, index4)

            index5 = self.Select_Expert_ByData(self.InverseMNISTTest_x)
            IMNISTError = self.Calculate_Elbo_ByIndex(self.InverseMNISTTest_x, index5)

            sum1 = mnistError + fashionError + svhnError + IFashionError + IMNISTError
            sum1 = sum1 / 5.0

            print(mnistError)
            print('\n')
            print(svhnError)
            print('\n')
            print(fashionError)
            print('\n')
            print(IFashionError)
            print('\n')
            print(IMNISTError)
            print('\n')
            print(sum1)

            for i in range(self.componentCount):
                gen = self.VAE_GeneratorList[i]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                g1 = sess.run(gen,feed_dict={self.z:batch_z})
                ims("results/" + "GANs" + str(i) + ".png", merge2(g1[:64], [8, 8]))

    def Select_Expert_ByData(self,textX):
        myCount = 1000
        realImages = textX[0:myCount]

        # Calculate FID
        fidArr = []
        for tIndex in range(self.componentCount):
            myIndex = tIndex + 1
            fakeImages = []
            realImages_set = []
            myGANs = self.SelectVAEs_Generator_byIndex(myIndex)
            myFeatures = self.SelectVAEs_Features_byIndex(myIndex)
            tt = int(myCount / self.batch_size)
            for i in range(tt):
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                aa = self.sess.run(myGANs, feed_dict={self.z: batch_z})

                batch_realSamples = realImages[i * self.batch_size:(i + 1) * self.batch_size]

                fff = self.sess.run(myFeatures, feed_dict={self.inputs: aa})
                fff2 = self.sess.run(myFeatures, feed_dict={self.inputs: batch_realSamples})
                for j in range(self.batch_size):
                    fakeImages.append(fff[j])
                    realImages_set.append(fff2[j])

            fakeImages = np.array(fakeImages)
            realImages_set = np.array(realImages_set)

            score = compute_mmd(fakeImages, realImages_set)
            print(score)
            fidArr.append(score)

        # Compare FID
        minIndex = fidArr.index(min(fidArr))
        minFid = min(fidArr)
        minIndex = minIndex + 1

        return minIndex

    def Calculate_Elbo_ByIndex2(self, testX,testY, index):
        p1 = int(np.shape(testX)[0] / self.batch_size)
        myPro = self.Elbo_List[index]
        sumError = 0
        for i in range(p1):
            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
            g = testX[i * self.batch_size:(i + 1) * self.batch_size]
            b = testY[i * self.batch_size:(i + 1) * self.batch_size]
            sumError = sumError + self.sess.run(myPro, feed_dict={self.inputs: g,self.labels:b,self.z:batch_z})

        sumError = sumError / p1
        return sumError

    def Select_Expert_ByData2(self,textX,testY):
        myCount = 1000
        realImages = textX

        # Calculate FID
        fidArr = []
        for tIndex in range(self.componentCount):
            myIndex = tIndex + 1
            score = self.Calculate_Elbo_ByIndex2(textX,testY,tIndex)
            fidArr.append(score)

        # Compare FID
        minIndex = fidArr.index(min(fidArr))
        minFid = min(fidArr)
        minIndex = minIndex + 1

        return minIndex

    def SelectVAEs_byIndex(self, index):
        return self.VAE_List[index - 1]

    def SelectVAEs_Generator_byIndex(self, index):
        return self.VAE_GeneratorList[index - 1]

    def SelectVAEs_Features_byIndex(self, index):
        return self.Feature_List[index - 1]

    def GenerateSamplesBySelect(self, n, index):
        myGANs = self.SelectVAEs_Generator_byIndex(index)
        a = int(n / self.batch_size)
        myArr = []
        for i in range(a):
            batch_labels = np.random.multinomial(1,
                                                 10 * [float(1.0 / 10.0)],
                                                 size=[self.batch_size])
            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
            aa = self.sess.run(myGANs, feed_dict={self.z: batch_z,self.labels:batch_labels})
            for t in range(self.batch_size):
                myArr.append(aa[t])
        myArr = np.array(myArr)
        return myArr

    def Predictions_By_Index(self,testX,index):
        myPredict = self.classifierPredictionList[index - 1]

        totalN = np.shape(testX)[0]
        myN = int(totalN / self.batch_size)
        myPrediction = myPredict
        totalPredictions = []
        myCount = 0
        for i in range(myN):
            my1 = testX[self.batch_size * i:self.batch_size * (i + 1)]
            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
            predictions = self.sess.run(myPrediction, feed_dict={self.inputs: my1,self.z:batch_z})
            for k in range(self.batch_size):
                totalPredictions.append(predictions[k])

        totalPredictions = np.array(totalPredictions)
        totalPredictions = keras.utils.to_categorical(totalPredictions, 10)
        return totalPredictions

    def Give_Reconstruction_ByIndex(self,samples,index):
        myReco = self.VAE_List[index-1]
        a = self.sess.run(myReco,feed_dict={self.inputs:samples})
        return a

    def Calculate_FID_Score(self, nextTaskIndex):

        nextTrainX = self.x_train
        nextTrainX = np.reshape(nextTrainX,(-1,28*28))
        nextTrainX = nextTrainX[:,self.task_permutation[nextTaskIndex]]
        nextTrainX = np.reshape(nextTrainX,(-1,28,28,1))
        nextTrainY = self.y_train

        myCount = 1000
        realImages = nextTrainX[0:myCount]
        realLabels = nextTrainY[0:myCount]

        # Calculate FID
        fidArr = []
        for tIndex in range(self.componentCount):
            myIndex = tIndex + 1
            fakeImages = []
            fakeLabels = []
            realImages_set = []
            myGANs = self.SelectVAEs_Generator_byIndex(myIndex)
            tt = int(myCount / self.batch_size)
            for i in range(tt):
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                batch_labels = np.random.multinomial(1,
                                                     10 * [float(1.0 / 10.0)],
                                                     size=[self.batch_size])
                aa = self.sess.run(myGANs, feed_dict={self.z: batch_z,self.labels:batch_labels})

                batch_realSamples = realImages[i * self.batch_size:(i + 1) * self.batch_size]
                for j in range(self.batch_size):
                    fakeImages.append(aa[j])
                    fakeLabels.append(batch_labels[j])

            otherElbo = self.Calculate_Elbo_ByIndex2(realImages,realLabels, tIndex)
            fakeImages = np.array(fakeImages)
            elbo = self.Calculate_Elbo_ByIndex2(fakeImages,fakeLabels, tIndex)
            diff = np.abs(otherElbo - elbo)
            fidArr.append(diff)
            print(diff)

        # Compare FID
        minIndex = fidArr.index(min(fidArr))
        minFid = min(fidArr)
        minIndex = minIndex + 1

        return minIndex, minFid

    def train(self):

        taskCount = 4

        config = tf.ConfigProto(allow_soft_placement=False)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        config.gpu_options.allow_growth = True

        self.componentCount = 1
        self.currentComponent = 1
        self.fid_hold = 200
        self.IsAdd = 0

        num_tasks_to_run = 10

        isFirstStage = True
        with tf.Session(config=config) as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()


            # self.saver.restore(sess, 'models/TeacherStudent_MNIST_TO_Fashion_invariant')

            # saver to save model
            self.saver = tf.train.Saver()
            ExpertWeights = np.ones((self.batch_size, 4))

            DomainState = np.zeros(4).astype(np.int32)
            DomainState[0] = 0
            DomainState[1] = 1
            DomainState[2] = 2
            DomainState[3] = 3

            taskCount = 5
            for taskIndex in range(num_tasks_to_run):

                print(np.shape(self.x_train))
                currentTrainX = self.x_train
                currentTrainX = np.reshape(currentTrainX,(-1,28*28))
                currentTrainX = currentTrainX[:, self.task_permutation[taskIndex]]
                currentTrainX = np.reshape(currentTrainX,(-1,28,28,1))
                currentTrainY = self.y_train

                if self.IsAdd == 0:
                    if taskIndex != 0:
                        oldX = self.GenerateSamplesBySelect(50000, self.currentComponent)
                        oldY = self.Predictions_By_Index(oldX,self.currentComponent)
                        oldX = oldX[0:np.shape(oldY)[0]]
                        currentTrainX = np.concatenate((currentTrainX, oldX), axis=0)
                        currentTrainY = np.concatenate((currentTrainY, oldY), axis=0)
                    '''
                    currentTrainX = self.cifar_train_x
                    currentTrainY = self.CifarTrain_y
                    currentTrain_labels = self.cifar_train_label
                    '''

                dataX = currentTrainX
                n_examples = np.shape(dataX)[0]

                start_epoch = 0
                start_batch_id = 0
                self.num_batches = int(n_examples / self.batch_size)

                mnistAccuracy_list = []
                mnistFashionAccuracy_list = []

                #self.epoch = 1
                # loop for epoch
                start_time = time.time()
                for epoch in range(start_epoch, self.epoch):
                    count = 0
                    # Random shuffling
                    index = [i for i in range(n_examples)]
                    random.shuffle(index)
                    dataX = dataX[index]
                    currentTrainY = currentTrainY[index]
                    counter = 0

                    # get batch data
                    for idx in range(start_batch_id, self.num_batches):
                        batch_images = dataX[idx * self.batch_size:(idx + 1) * self.batch_size]
                        batch_labels = currentTrainY[idx * self.batch_size:(idx + 1) * self.batch_size]

                        # update GAN
                        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                        dataIndex = [i for i in range(self.batch_size)]
                        random.shuffle(dataIndex)

                        # update G and Q network
                        _, vaeLoss,_,classifier_loss = self.sess.run(
                            [self.vae_optim, self.vaeLoss,self.classifier_optim,self.classLoss],
                            feed_dict={self.inputs: batch_images, self.z: batch_z,self.labels:batch_labels
                                      })

                        # display training status
                        counter += 1
                        print(
                            "Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, vae_loss:%.8f. c_loss:%.8f" \
                            % (epoch, idx, self.num_batches, time.time() - start_time, vaeLoss, classifier_loss, 0, 0))

                nextTaskIndex = taskIndex + 1

                if taskIndex < 9:
                    minIndex, minFID = self.Calculate_FID_Score(nextTaskIndex)

                    # minFID = 300

                    self.fid_hold = 100
                    print("Score")
                    if self.componentCount < 7:
                        self.fid_hold = 50

                    if minFID > self.fid_hold:  # add a new GANs
                        self.currentComponent = self.componentCount + 1
                        self.componentCount = self.componentCount + 1
                        self.IsAdd = 1
                        self.Create_NewExpert()
                        self.Activate_Expert(self.currentComponent)
                        print("Add")
                    else:
                        # continous to use the current GANs
                        self.IsAdd = 0
                        #print(minIndex)
                        self.currentComponent = minIndex
                        self.Activate_Expert(self.currentComponent)

                    print(self.componentCount)

            #Testing phase
            print("Number of components")
            print(self.componentCount)
            sum1 = 0
            for taskIndex in range(num_tasks_to_run):
                testx = self.x_test
                testx = np.reshape(testx,(-1,28*28))
                testx = testx[:,self.task_permutation[taskIndex]]
                testx = np.reshape(testx,(-1,28,28,1))
                testy = self.y_test
                index1 = self.Select_Expert_ByData2(testx, testy)
                mnistError = self.Calculate_Accuracy_ByIndex(testx, testy, index1)

                sum1 = sum1 + mnistError
            sum1 = sum1 / 10.0
            print("Accuracy")
            print(sum1)

            for i in range(self.componentCount):
                gen = self.VAE_GeneratorList[i]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                batch_labels = np.random.multinomial(1,
                                                     10 * [float(1.0 / 10.0)],
                                                     size=[self.batch_size])
                g1 = sess.run(gen, feed_dict={self.z: batch_z,self.labels:batch_labels})
                ims("results/" + "Supervised_shared_log_GANs" + str(i) + ".png", merge2(g1[:64], [8, 8]))

            #self.saver.save(self.sess, "models/InfiniteVAE_Share.ckpt")

infoMultiGAN = LifeLone_MNIST()
infoMultiGAN.build_model()
infoMultiGAN.train()
# infoMultiGAN.train_classifier()
#infoMultiGAN.test()
