""" Test Hessian Free Optimizer on XOR and MNIST datasets """
""" Author: MoonLight, 2018 """
""" Modified for TensorFlow 2 compatibility """

import numpy as np
import tensorflow as tf
from hfoptimizer import HFOptimizer
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

np.set_printoptions(suppress=True)

""" Run example on MNIST or XOR """
DATASET = 'MNIST'
class OneHotLayer(tf.keras.layers.Layer):
    def __init__(self, depth, **kwargs):
        super(OneHotLayer, self).__init__(**kwargs)
        self.depth = depth

    def call(self, inputs):
        return tf.one_hot(inputs, depth=self.depth, dtype=tf.float64)



def example_XOR():
    x = tf.keras.Input(shape=(2,), dtype=tf.float64, name='input')
    y = tf.keras.Input(shape=(1,), dtype=tf.float64, name='output')

    with tf.name_scope('ffn'):
        W_1 = tf.Variable([[3.0, 5.0], [4.0, 7.0]], dtype=tf.float64, name='weights_1')
        b_1 = tf.Variable(tf.zeros([2], dtype=tf.float64), name='bias_1')
        y_1 = tf.sigmoid(tf.matmul(x, W_1) + b_1)

        W_2 = tf.Variable([[-8.0], [7.0]], dtype=tf.float64, name='weights_2')
        b_2 = tf.Variable(tf.zeros([1], dtype=tf.float64), name='bias_2')
        y_out = tf.matmul(y_1, W_2) + b_2

        out = tf.nn.sigmoid(y_out)

    """ Log-loss cost function """
    loss = tf.reduce_mean(((y * tf.math.log(out)) +
                           ((1 - y) * tf.math.log(1.0 - out))) * -1, name='log_loss')

    XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    XOR_Y = [[0], [1], [1], [0]]

    model = tf.keras.Model(inputs=x, outputs=out)
    hf_optimizer = HFOptimizer(loss=loss, model=model, dtype=tf.float64)

    max_epochs = 100
    print('Begin Training')
    for i in range(max_epochs):
        hf_optimizer.minimize(model)
        if i % 10 == 0:
            print('Epoch:', i, 'cost:', loss(model(XOR_X), XOR_Y).numpy())
            print('Hypothesis:', model(XOR_X).numpy())

def example_MNIST():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10
    np.set_printoptions(suppress=True)

    """ Constructing simple neural network """
    inputs = tf.keras.Input(shape=(n_inputs,), dtype=tf.float64, name='input')
    x = tf.keras.layers.Flatten()(inputs)
    y_1 = tf.keras.layers.Dense(n_hidden1, activation='sigmoid', dtype=tf.float64)(x)
    y_out = tf.keras.layers.Dense(n_outputs, dtype=tf.float64)(y_1)
    model = tf.keras.Model(inputs=inputs, outputs=y_out)

    class SoftmaxCrossEntropyLoss(tf.keras.layers.Layer):
        def __init__(self, num_classes):
            super(SoftmaxCrossEntropyLoss, self).__init__()
            self.num_classes = num_classes

        def call(self, y_true, y_pred):
            y_true = tf.one_hot(y_true, self.num_classes, dtype=tf.float64)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            return tf.reduce_mean(loss)

    loss_fn = SoftmaxCrossEntropyLoss(n_outputs)

    @tf.function
    def compute_accuracy(y_pred, y_true):
        correct = tf.nn.in_top_k(tf.cast(y_pred, tf.float32), y_true, 1)
        return tf.reduce_mean(tf.cast(correct, tf.float64))

    n_epochs = 2
    batch_size = 50

    """ Initializing Hessian-free optimizer """
    hf_optimizer = HFOptimizer(loss=loss_fn, model=model, dtype=tf.float64, batch_size=batch_size)
    hf_optimizer.info()

    train_images = train_images.reshape((-1, n_inputs)).astype(np.float64) / 255.0
    test_images = test_images.reshape((-1, n_inputs)).astype(np.float64) / 255.0

    for epoch in range(n_epochs):
        n_batches = train_images.shape[0] // batch_size
        for iteration in range(n_batches):
            x_batch = train_images[iteration * batch_size:(iteration + 1) * batch_size]
            t_batch = train_labels[iteration * batch_size:(iteration + 1) * batch_size]
            hf_optimizer.minimize(model)

            if iteration % 1 == 0:
                print('Batch:', iteration, '/', n_batches)
                acc_train = compute_accuracy(model(x_batch), t_batch).numpy()
                acc_test = compute_accuracy(model(test_images), test_labels).numpy()
                print('Loss:', loss_fn(t_batch, model(x_batch)).numpy())
                print('T', t_batch[0])
                print('Out:', tf.nn.softmax(model(x_batch)).numpy()[0])
                print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

        acc_train = compute_accuracy(model(x_batch), t_batch).numpy()
        acc_test = compute_accuracy(model(test_images), test_labels).numpy()
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

if __name__ == '__main__':
    print('Running Hessian Free optimizer test on:', DATASET)
    if DATASET == 'MNIST':
        example_MNIST()
    elif DATASET == 'XOR':
        example_XOR()
    else:
        print(bcolors.FAIL +
              'Unknown DATASET parameter, use only XOR or MNIST' + bcolors.ENDC)