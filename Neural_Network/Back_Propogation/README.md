# Back Propogation

Back Propogation is the algorithm to train the neural network  by adjusting the weights of the neurons in response to the error between the predicted output and the actual output.

## Steps
1. Intialize weights and biases (Randomly or weights as 1 and biases as 0)
2. Run a loop for no. of epoches
3. In the above loop run another loop for training examples in the datasets:
    - forward propogation for each row, i.e., predict the output  using current weights and biases
    - Calculate the loss using the mentioned loss function
    - Adjust the weights and biases using back propagation algorithm:
        - W_new = W_old - lr * (d(Loss)/dW)
        - B_new = B_old - lr * d(Loss)/dB
4. Print the accuracy after every n number of iterations/epoches

- Link to visualize Back Prop - [Back Propagation](https://developers-dot-devsite-v2-pro...)


## Memoization in Back Prop

Memoization -> Memoization is a technique in computer science where you store the results of expensive function calls and reuse those results when the same inputs occur again, instead of recalculating the function. This can significantly improve the performance of recursive functions or functions that have repeated calls with the same inputs.

- Derivatives are stored in the memory when propogating backwards as they are required as we go back in the network.

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

## improvr nn 
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
