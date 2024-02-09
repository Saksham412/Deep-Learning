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
