# Forward Prpogation

Forward propagation is the process of calculating the output of a neural network given a set of inputs the network's weights, and biases.

## Overview

Forward propagation consists of the following steps:

1. Initialization: Set the initial values for the network's weights and biases. These values can be randomly initialized or set using a technique such as Xavier initialization.
2. Input Layer: Pass the input data through the input layer of the network. This involves multiplying the input data by the weights of the input layer and adding the biases.
3. Hidden Layers: Pass the output of the input layer through each of the hidden layers in the network. This involves calculating the weighted sum of the inputs to each neuron, applying the activation function, and adding the biases.
4. Output Layer: Pass the output of the final hidden layer through the output layer of the network. This involves calculating the weighted sum of the inputs to each neuron and applying the activation function.
5. Prediction: The output of the output layer is the network's prediction for the given input data.

## Mathematical Formulation

The forward propagation process can be mathematically represented as follows:

- Input layer:
z[1] = w[1] * x + b[1]

- Hidden layers:
a[l] = f(z[l]),  z[l+1] = w[l+1] * a[l] + b[l+1]

- Output layer:
y = f(z[L])

where:

- x is the input data
- w is the weights of the layer
- b is the biases of the layer
- a is the output of the layer (before applying the activation function)
- z is the weighted sum of the inputs to a neuron
- f is the activation function
- L is the number of layers in the network