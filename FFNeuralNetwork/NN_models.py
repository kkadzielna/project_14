from keras.models import Sequential
from keras.layers import Dense

import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dropout

##Start off with a base model of 3 and


def NN1_Base_Model_Build(layer_number, activation, in_shape, classes, output_activation, optimizer, loss):

    model = models.Sequential()

    ##Input Layer
    model.add(Dense(36, activation = activation, input_shape = in_shape))

    for i in range(layer_number):

        ##Hidden Layers
        model.add(Dense(36, activation = activation))


    #Output Layer
    model.add(Dense(classes, activation = output_activation))

    #Compiling
    model.compile(optimizer = optimizer,
                  loss = loss,
                  metrics = ['acc'])

    return model

def finding_best_neurons_combo(neuron, activation, in_shape, classes, output_activation, optimizer, loss):

    model = models.Sequential()

    ##Input Layer
    model.add(Dense(37, activation = activation, input_shape = in_shape))

    #Hidden Layers
    for i in neuron:

        model.add(Dense(i, activation = activation))

    ##Ouput Layer
    model.add(Dense(classes, activation = output_activation))

    ##Compiling
    model.compile(optimizer = optimizer,
                  loss = loss,
                  metrics = ['acc'])

    return model

def NN_Model_Dropout(neuron, activation, in_shape, classes, output_activation, optimizer, loss, dr):

    model = models.Sequential()

    ##Input Layer
    model.add(Dense(37, activation = activation, input_shape = in_shape))

    #Hidden Layers
    for i in neuron:

        model.add(Dense(i, activation = activation))
        model.add(Dropout(dr))

    ##Ouput Layer
    model.add(Dense(classes, activation = output_activation))

    ##Compiling
    model.compile(optimizer = optimizer,
                  loss = loss,
                  metrics = ['acc'])

    return model


def NN1_Model_Build(activation, in_shape, classes, output_activation, optimizer, loss):

  model = Sequential([
                      Dense(37, activation = activation, input_shape = in_shape),
                      Dense(classes, activation = output_activation)
                      ])

  model.compile(optimizer = optimizer,
              ##loss = 'categorical_crossentropy',
              loss = loss,
              metrics = ['acc'])

  return model

def NN2_Model_Build(activation, in_shape, classes, output_activation, optimizer, loss):

  model = Sequential([
                      Dense(256, activation = activation, input_shape = in_shape),
                      Dense(128, activation = activation),
                      Dense(64, activation = activation),
                      Dense(13, activation = output_activation),
                      ])

  model.compile(optimizer = optimizer,
              ##loss = 'categorical_crossentropy',
              loss = loss,
              metrics = ['acc'])

  return model
