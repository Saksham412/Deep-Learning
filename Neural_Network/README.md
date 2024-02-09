# Multi Layer Perceptron (Neural Network)

- Check out different architectures and play with neural networks on following link:
[Tensorflow Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.96653&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

## Number of Parameters
The number of parameters in a neural network is determined by the architecture of the network.

![Perceptron](../images/neural_network.png)

    Given network contains:
        4 Layers, 2 Hidden Layers with 4 perceptron each, 1 Output Layer with 1 neuron and 1 Iutput Layer with 3 perceptron.

        - No. of parameters to be measured is calculated by multiplying the no.of input perceptrons to output perceptrons and adding biases of each output perceptron.
        For example - [(3*4)+4] + [(4*4)+4] + [(4*1)+1] = [16]+[20]+[5]= 47 neurons in total
        
        The first layer is connected to the second one by a fully-connected connection, which means that each neuron of the first layer is connected with all neurons of the second layer

## Notations

W^(layer_no)_(input_neuron_no)(output_neuron_no)

For the above network:
    W^2_14->Weight of the 1'st neuron of 2'nd layer connecting to 4'th neuron  of 3'rd layer.
    b_21-> Bias of the 2'nd layer's 1'st neuron.
    

# Loss Functions in Neural Network

Loss functions are used to evaluate how well the model performs its task. There are several loss functions available mainly:

    1.Mean Squared Error (MSE): (y'-y)^2 
        - Advantages: Easy to interpret, differentiable, 1 local minima
        - Disadvantages: Error unit (squared) -> differentiable, Not robust to outliers that is it punishes the outliers.

    2.Mean Absolute Error (MAE): |y'-y|
        - Advantages: Easy to interpret, intuitive, Robust to outliers, unit same
        - Disadvantages: Error unit (squared) -> not differentiable

    3.Huber loss: (if |y-y'|<=delta then H=(y-y')^2/2 else H=delta*|y-y'|-(delta^2)/2)
        - Works well when there more outliers in the data

    4.Binary crossentropy loss or Log loss: -[(1-y')*log(1-y')+ (y' * log y') ]
        - Used for binary classification problems
        - Activation function at the output layer should be sigmoid.

    5.Categorical Cross Entropy Loss: -(1/n)*sum(y'_i *log(y))
        - Used for multi class classification problem
        - Softmax activation function used at the output layer.
        - Neurons at the output layer should be as many as classed.
        - We need to onehotencode the data before applying this loss function else we can use Sparse Categorical cross emtropy.
        - Sparse Categorical cross emtropy is prefereed as we dont need to onehotencode and it is faster for many class dataset.
    
**NOTE**: Cost function is addition of all rows loss in the dataset whereas loss function is for only one particular row.


## Gradient Descent Algorithms

### Batch Gradient Descent
- Uses the entire dataset to compute the gradient.
- Updates model parameters once per epoch.
- More computationally expensive but converges smoothly.

### Stochastic Gradient Descent (SGD)
- Uses only one randomly chosen data point to compute the gradient.
- Updates model parameters after each data point.
- Less computationally expensive but may exhibit more oscillations during convergence.

### Mini batch gradient descent 
- Uses a small random subset (mini-batch) of the dataset to compute the gradient.
- Updates model parameters after processing each mini-batch.
- Strikes a balance between the smooth convergence of batch gradient descent and the computational efficiency of SGD.


## Vanishing and Exploding gradient problem

### Vanishing Gradient Problem
- Occurs when gradients become extremely small during backpropagation.
- Common in deep neural networks with many layers, especially those using activation functions like sigmoid or tanh.
- Leads to slow or stalled learning in early layers, hindering model convergence.
- Mitigated by using activation functions like ReLU and careful initialization of network weights.

### Exploding Gradient Problem
- Occurs when gradients become extremely large during backpropagation.
- Can cause unstable training, diverging loss, and parameter updates that oscillate wildly.
- Often observed in deep networks with very large learning rates or poorly conditioned optimization problems.
- Remedied by gradient clipping, reducing learning rates, or using techniques like batch normalization.

## Improving Neural Network 
- HyperParamterTuning:
    
    1. **Learning Rate**: Determines the step size during gradient descent.

    2. **Number of Hidden Layers and Neurons**: Impact the network's capacity to learn complex patterns.

    3. **Activation Functions**: Choices like ReLU, tanh, or sigmoid influence the network's ability to capture nonlinear relationships.

    4. **Regularization**: Techniques like L1/L2 regularization, dropout, or batch normalization prevent overfitting.

    5. **Batch Size**: Determines the number of training examples processed before updating model parameters.

    6. **Initialization Schemes**: Techniques like Xavier/Glorot initialization ensure appropriate weight initialization.

    7. **Optimizer Choice**: Algorithms like SGD, Adam, or RMSprop determine how model parameters are updated.

    8. **Learning Rate Schedule**: Techniques like learning rate decay adjust the learning rate during training.

    9. **Validation Strategy**: Determines how model performance is evaluated during hyperparameter search.

    10. **Early Stopping**: Prevents overfitting by stopping training when validation performance plateaus.

    11. **Hyperparameter Search Strategy**: Methods like grid search, random search, or Bayesian optimization explore the hyperparameter space efficiently.

    12. **Model Architecture Modifications**: Techniques like adding skip connections or using different layer types can improve performance.

- Addressing Common Challenges in Neural Networks

    1. Vanishing Gradient Problem
        - **Issue**: Gradients become extremely small during backpropagation, leading to slow learning in early layers.
        - **Solution**: Use activation functions like ReLU instead of sigmoid or tanh to mitigate vanishing gradients. Proper weight initialization and batch normalization can also help.

    2. Overfitting
        - **Issue**: Model learns to memorize training data rather than generalize, leading to poor performance on unseen data.
        - **Solution**: Employ regularization techniques like L1/L2 regularization, dropout, or early stopping. Increase dataset size or use data augmentation to provide more diverse examples.

    3. Insufficient Data
        - **Issue**: Not enough data to effectively train the model, leading to poor generalization.
        - **Solution**: Use techniques like transfer learning, where a pre-trained model on a larger dataset is fine-tuned on the target dataset. Apply data augmentation to artificially increase the dataset size. Utilize techniques like semi-supervised learning or generative adversarial networks (GANs) to generate synthetic data.

    4. Slow Training
        - **Issue**: Training the model takes too long, hindering development and experimentation.
        - **Solution**: Utilize hardware acceleration (e.g., GPUs, TPUs) to speed up training. Optimize network architecture and hyperparameters to reduce computational complexity. Implement distributed training across multiple devices or use techniques like model distillation.

- **Early Stopping in Neural Networks** - 

    Early stopping is a technique used to prevent overfitting in neural networks by monitoring the model's performance on a validation dataset during training and stopping the training process when the performance starts to degrade.

    In Keras, early stopping can be implemented using the `EarlyStopping` callback, which is part of the `keras.callbacks` module. This callback allows you to specify various parameters to control the early stopping behavior, such as the monitoring metric (e.g., validation loss or accuracy), the minimum change required to qualify as an improvement (`min_delta`), and the patience, which determines the number of epochs with no improvement after which training will be stopped.

    ```python
    from keras.callbacks import EarlyStopping
    from keras.models import Sequential
    from keras.layers import Dense

    # Define the model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(10,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Define early stopping criteria
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    # Train the model with early stopping
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), callbacks=[early_stopping])

