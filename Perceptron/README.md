# Perceptron

A perceptron is the simplest form of a neural network, serving as a basic building block for more complex models. It's often used for binary classification tasks.

![Perceptron](../images/perceptron.png)

## Comparison with Neuron

- **Complexity**: A single neuron in a biological brain is highly complex, involving various biological processes. In contrast, a perceptron is a simplified mathematical model, consisting of input features, weights, and an activation function.
  
- **Processing**: Neurons in the brain process information in parallel, forming intricate networks. Perceptrons, on the other hand, process inputs sequentially, updating weights based on input signals and predefined rules.
  
- **Neuroplasticity**: Biological neurons exhibit neuroplasticity, allowing them to adapt and reorganize connections based on experience. Perceptrons lack this capability and require manual adjustments to their parameters for learning.

Weight values in a perceptron indicate the importance of features in the classification process.

## Geometric Intuition

The perceptron classifies data into two classes by creating a decision boundary, which could be a line in 2D, a plane in 3D, or a hyperplane in higher dimensions. This decision boundary separates the classes based on the features of the input data.

# Training Perceptron 

## Perceptron Trick
- We take a line and run a loop till convergence or for 1000 epoches, for each interval we select a point and check wheather the point is at correct position and update the line accordingly
    - Now to update the line for a particular point we take some learning rate like 0.01 or .01 depending on the data and multiply the coordinates and subtract this from the coef of the line (learning rate is because of the high transformation in line otherwise)

- **Algorithm** - 
```
for i in range (epochs):
    // select a random student i 
    w_n = w_0+n*(y_i-y_i_^)*x_i
```
## Problem with perceptron

- line may change each time we train the perceptron
- convergence may not occur each time

# Perceptron Loss Function

```
import numpy as np

def perceptron_loss(y, y_pred, weights, learning_rate):
    """
    Calculates the perceptron loss a given set of true labels y, predicted labels y_pred, weights, and learning_rate.

    :param y: numpy array of true labels with shape (n_samples,)
    :param y_pred: numpy array of predicted labels with shape (n_samples,)
    :param weights: numpy array of weights with shape (n_features,)
    :param learning_rate: float, learning rate for the perceptron
    :return: float, the perceptron loss
    """
    n_samples = len(y)
    loss = 0.0

    # Calculate loss for each sample
    for i in range(n_samples):
        x_i = np.array([1] + list(y_pred[i]))  # Add bias term
        error = y[i] - np.sign(np.dot(x_i, weights))
        loss += max(0, error)

        # Update weights using the perceptron trick
        weights += learning_rate * error * x_i

    return loss
```
This function calculates the perceptron loss for a given set of true labels y, predicted labels y_pred, weights weights, and learning rate learning_rate. It also updates the weights using the perceptron trick during the loss calculation.

## Gradient descent 

Gradient descent is not directly applicable to the perceptron loss function due to its non-differentiability at the decision boundary. Instead, the perceptron algorithm uses the perceptron trick, which is a form of online gradient descent that updates the weights sequentially based on the sign of the error.
 
# Important points to note about Perceptron

-It can be used fo different algorithms by using different activation functions and different loss functins.
    - Perceptron or Binary classifier -> Activation function = step and Loss function = Hinge loss
    - Logistic regression -> Activation function = sigmoid and Loss function = Binary cross entropy
    - Linear regression -> Activation function = Linear and Loss function = mean squared error
    - maulticlass classification -> Activation function = softmax and Loss function = categorical cross entropy etc



