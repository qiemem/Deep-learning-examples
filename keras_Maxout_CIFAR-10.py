#!/usr/bin/env python3

description = """
## Maxout on CIFAR-10

This script demonstrates a "maxout" type NN as presented in

[Goodfellow, I. J., Warde-Farley, D., Mirza, M., Courville, A. C., & Bengio, Y. (2013). Maxout Networks. ICML'13, 28, 1319-1327.](http://www-etud.iro.umontreal.ca/~goodfeli/maxout.html)

using Keras.

The network structure was mostly copied from https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/scripts/papers/maxout/cifar10_no_aug.yaml (which corresponds with the paper), though not all preprocessing, regularization, or optimizer settings are the same. As a result of these differences, the performance of this network is somewhat worse than what is presented in the paper.
"""


import keras
from keras.datasets import cifar10
from keras.models import Model
from keras import layers
from keras import optimizers
from keras import backend as K

import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def load_cifar():
    """
    Load the CIFAR dataset and normalizes it to 0-1 float32s.
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return ((x_train, y_train), (x_test, y_test))


def create_maxout_model(input_shape):
    """
    Create a maxout model with three convolutional maxout layers and one dense
    maxout layer.

    Args:
        input_shape (tuple):
            shape of the images to run on; i.e. (rows, cols, channels)
    Returns:
        the compiled keras model, ready to be trained.
    """

    inputs = layers.Input(shape = input_shape, name='input')

    x = layers.Dropout(0.2, name='dropout_1')(inputs)

    # First maxout layer
    x = layers.Maximum(name='maxout_1')([
        layers.Conv2D(96, (8,8), padding = 'same', name='conv_1_{}'.format(i))(x)
        for i in range(2)
    ])
    x = layers.MaxPool2D(name='maxpool_1')(x)

    x = layers.Dropout(0.2, name='dropout_2')(x)

    # Second maxout layer
    x = layers.Maximum(name='maxout_2')([
        layers.Conv2D(192, (8,8), padding = 'same', name='conv_2_{}'.format(i))(x)
        for i in range(2)
    ])
    x = layers.MaxPool2D(name='maxpool_2')(x)

    x = layers.Dropout(0.2, name='dropout_3')(x)

    # Third maxout layer
    x = layers.Maximum(name='maxout_3')([
        layers.Conv2D(192, (5,5), padding = 'same', name='conv_3_{}'.format(i))(x)
        for i in range(2)
    ])
    x = layers.MaxPool2D(name='maxpool_3')(x)

    x = layers.Flatten(name='flatten')(x)

    x = layers.Dropout(0.2, name='dropout_4')(x)

    # Dense maxout layer
    x = layers.Maximum(name='maxout_5')([
        layers.Dense(500, name='dense_1_{}'.format(i))(x)
        for i in range(5)
    ])

    x = layers.Dropout(0.2, name='dropout_5')(x)
    predictions = layers.Dense(10, activation='softmax', name='dense_2')(x)

    model = Model(inputs = inputs, outputs = predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer = optimizers.adadelta(),
                  metrics=['accuracy'])
    return model


def train_maxout_model(model = None, **kwargs):
    """
    Train the model on the CIFAR10 dataset.

    Args:
        model: If None, a newly generated Maxout model is trained. Otherwise, the
            given model is used.
        **kwargs: All keyword arguments are passed on to model.fit. See Keras
            documentation.

    Returns:
        (model, history)
    """

    ((x_train, y_train), (x_test, y_test)) = load_cifar()

    if model == None:
        model = create_maxout_model(x_train.shape[1:])

    actual_kwargs = dict({
        'batch_size': 128,
        'epochs': 50,
        'verbose': 1,
        'validation_data': (x_test, y_test)
    }, **kwargs)

    if actual_kwargs['verbose'] > 0:
        model.summary()
    history = model.fit(x_train, y_train, **actual_kwargs)
    if actual_kwargs['verbose'] > 0:
        score = model.evaluate(x_test, y_test, batch_size = actual_kwargs['batch_size'])
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    return model, history

if __name__ == '__main__':
    parser = ArgumentParser(description, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log',
                        help='CSV file to store training progress to.')
    parser.add_argument('-v', '--verbose', nargs='?', type=int, default=2, const=1,
                        help='0 = silent, 1 = verbose, 2 = one log line per epoch.')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Number of samples in each batch.')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of epochs to train for.')
    parser.add_argument('-s', '--save',
                        help='HDF5 file to save the learned weights in. If none is specified, the weights are not saved.')
    parser.add_argument('-l', '--load', help='Use the model in the specified HDF5 file.')
    parser.add_argument('--initial_epoch', type=int, default=0, help='Which epoch number to start at. Use if continuing training on existing model.')

    args = parser.parse_args()

    loaded_model = None if args.load == None else keras.models.load_model(args.load)

    callbacks = [] if args.log == None else [keras.callbacks.CSVLogger(args.log)]

    model, history = train_maxout_model(
        loaded_model,
        verbose = args.verbose,
        batch_size=args.batch_size,
        epochs = args.epochs,
        initial_epoch = args.initial_epoch,
        callbacks = callbacks
    )

    if args.save != None:
        model.save(args.save)


