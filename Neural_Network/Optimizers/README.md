- **Challanges:** 
1. Learning rate initialsation
2. Learning rate scheduling
3. separte learning rate for each dimension
4. Local minima
5. Saddle point

# Exponentially weighted moving average(EWMA)

Exponentially Weighted Moving Average (EWMA) is a statistical method used to smooth out time series data by giving more weight to recent observations while gradually decreasing the weight of older observations. It is widely used in finance, engineering, and signal processing to capture trends and identify patterns in data.

- **Formula:**
The formula for calculating the EWMA at time t, denoted as St, with a smoothing factor α and the time series data xt, is:

St = α × xt + (1 - α) × St-1

Where:
- xt = the value of the time series at time t.
- α = the smoothing factor, typically between 0 and 1. A smaller α gives more weight to recent observations, while a larger α gives more weight to older observations.
- St = the EWMA at time t.
- St-1 = the previous EWMA value.

The initial value of St is typically set to the first observation in the time series or an average of the initial observations.

The formula can also be expressed recursively.

## Usage
1. Provide the time series data xt.
2. Choose a suitable value for the smoothing factor α.
3. Use the formula to calculate the EWMA at each time point.

# Stochastic Gradient Descent (SGD) with Momentum

## Overview
Stochastic Gradient Descent (SGD) with Momentum is an optimization algorithm commonly used in training neural networks and other machine learning models. It is an extension of the standard SGD algorithm that incorporates momentum to accelerate convergence and improve stability.

## Algorithm
SGD with Momentum updates the model parameters θ at each iteration using a combination of the current gradient and a momentum term. The update rule for parameter θ is as follows:

v_t = β * v_{t-1} + (1 - β) * ∇θJ(θ)

θ = θ - α * v_t

Where:
- v_t is the momentum term at iteration t.
- β is the momentum hyperparameter, typically a value between 0 and 1.
- ∇θJ(θ) is the gradient of the objective function J(θ) with respect to the parameters θ.
- α is the learning rate.

The momentum term v_t accumulates gradients over time, allowing the optimization process to maintain momentum in directions with consistent gradients and dampening oscillations in directions with high variance.

## Usage
1. Choose a suitable value for the momentum hyperparameter β.
2. Initialize the momentum term v_0 to zero.
3. Iterate through the training data in mini-batches.
4. Compute the gradient of the objective function with respect to the parameters.
5. Update the momentum term and model parameters using the update rule.
6. Repeat steps 3-5 until convergence or a predefined number of iterations.

## Benefits
- **Faster convergence**: Momentum helps accelerate convergence by allowing the optimization process to build up speed in directions with consistent gradients.
- **Improved stability**: Momentum dampens oscillations in the optimization process, leading to more stable updates and smoother convergence trajectories.
- **Escape local minima**: Momentum can help escape local minima by allowing the optimization process to overcome small gradients and explore other regions of the parameter space.

## Example
```python
# Pseudocode for SGD with Momentum
v = 0
for each iteration t:
    gradient = compute_gradient(data, parameters)
    v = beta * v + (1 - beta) * gradient
    parameters = parameters - learning_rate * v


# Nesterov Accelerated Gradient (NAG)

## Overview
Nesterov Accelerated Gradient (NAG) is an optimization algorithm commonly used in training neural networks and other machine learning models. It is an extension of SGD with Momentum that aims to further improve convergence speed and accuracy by taking into account future gradient information.

## Algorithm
NAG modifies the standard SGD with Momentum update rule to include a lookahead step, where the gradient is evaluated ahead in the direction of the momentum. The update rule for parameter θ is as follows:

v_t = β * v_{t-1} + α * ∇θJ(θ - β * v_{t-1})

θ = θ - v_t

Where:
- v_t is the momentum term at iteration t.
- β is the momentum hyperparameter, typically a value between 0 and 1.
- ∇θJ(θ - β * v_{t-1}) is the gradient of the objective function J(θ) evaluated at the lookahead point.
- α is the learning rate.

NAG first takes a step in the direction of the previous momentum (β * v_{t-1}) and then updates the momentum term and model parameters using the gradient evaluated at this lookahead point. This lookahead step helps NAG to better anticipate the next update and improves convergence.

## Usage
1. Choose a suitable value for the momentum hyperparameter β.
2. Initialize the momentum term v_0 to zero.
3. Iterate through the training data in mini-batches.
4. Compute the gradient of the objective function with respect to the parameters at the lookahead point.
5. Update the momentum term and model parameters using the update rule.
6. Repeat steps 3-5 until convergence or a predefined number of iterations.

## Benefits
- **Faster convergence**: NAG leverages lookahead information to anticipate future updates, leading to faster convergence compared to standard SGD with Momentum.
- **Improved accuracy**: By incorporating lookahead steps, NAG can make more informed updates and achieve higher accuracy on the objective function.
- **Better handling of high curvature**: NAG is particularly effective in scenarios with high curvature or sharp turns in the optimization landscape, where standard SGD with Momentum may struggle.

## Example
```python
# Pseudocode for Nesterov Accelerated Gradient (NAG)
v = 0
for each iteration t:
    lookahead_gradient = compute_gradient(data, parameters - beta * v)
    v = beta * v + alpha * lookahead_gradient
    parameters = parameters - v


# AdaGrad

## Overview
AdaGrad (Adaptive Gradient Algorithm) is an optimization algorithm commonly used in training neural networks and other machine learning models. It adapts the learning rate for each parameter based on the historical gradients, allowing for larger updates for infrequent parameters and smaller updates for frequent parameters.

## Algorithm
AdaGrad maintains a per-parameter learning rate that is inversely proportional to the square root of the sum of squared gradients for that parameter up to the current iteration. The update rule for parameter θ is as follows:

g_t = ∇θJ(θ)

s_t = s_{t-1} + g_t^2

θ = θ - (α / sqrt(s_t + ε)) * g_t

Where:
- g_t is the gradient of the objective function J(θ) with respect to the parameters θ at iteration t.
- s_t is a diagonal matrix where each diagonal element is the sum of the squared gradients up to iteration t.
- α is the initial learning rate.
- ε is a small constant (usually on the order of 1e-8) added for numerical stability.

AdaGrad adapts the learning rate for each parameter based on its historical gradients. Parameters with larger gradients will have smaller effective learning rates, while parameters with smaller gradients will have larger effective learning rates. This allows AdaGrad to automatically adjust the learning rate to the specific requirements of each parameter during training.

## Usage
1. Choose a suitable initial learning rate α.
2. Initialize the sum of squared gradients s_0 to zero.
3. Iterate through the training data in mini-batches.
4. Compute the gradient of the objective function with respect to the parameters.
5. Update the sum of squared gradients and model parameters using the update rule.
6. Repeat steps 3-5 until convergence or a predefined number of iterations.

## Benefits
- **Adaptive learning rates**: AdaGrad adapts the learning rate for each parameter based on its historical gradients, allowing for larger updates for infrequent parameters and smaller updates for frequent parameters.
- **Automatic scaling**: AdaGrad automatically scales the learning rates based on the gradients, eliminating the need for manual tuning of learning rates.
- **Effective for sparse data**: AdaGrad performs well on sparse data where certain features have infrequent updates, as it adapts the learning rates based on the frequency of parameter updates.

## Example
```python
# Pseudocode for AdaGrad
s = 0
for each iteration t:
    gradient = compute_gradient(data, parameters)
    s = s + gradient^2
    parameters = parameters - (alpha / sqrt(s + epsilon)) * gradient

# RMSprop

## Overview
RMSprop (Root Mean Square Propagation) is an optimization algorithm commonly used in training neural networks and other machine learning models. It is designed to address the diminishing learning rates and exploding gradients problems often encountered in training deep neural networks.

## Algorithm
RMSprop adapts the learning rate for each parameter based on the magnitude of recent gradients. It maintains a moving average of the squared gradients for each parameter and scales the learning rate by the square root of this moving average. The update rule for parameter θ is as follows:

g_t = ∇θJ(θ)

E[g^2]_t = β * E[g^2]_{t-1} + (1 - β) * g_t^2

θ = θ - (α / sqrt(E[g^2]_t + ε)) * g_t

Where:
- g_t is the gradient of the objective function J(θ) with respect to the parameters θ at iteration t.
- E[g^2]_t is the moving average of the squared gradients for each parameter up to iteration t.
- β is the momentum decay parameter, typically a value close to 1 (e.g., 0.9).
- α is the learning rate.
- ε is a small constant (usually on the order of 1e-8) added for numerical stability.

RMSprop scales the learning rate based on the magnitude of recent gradients, effectively reducing the learning rate for parameters with large gradients and increasing it for parameters with small gradients. This helps stabilize the training process and prevent exploding gradients, leading to faster convergence and improved performance.

## Usage
1. Choose suitable values for the momentum decay parameter β and the initial learning rate α.
2. Initialize the moving average of squared gradients E[g^2]_0 to zero.
3. Iterate through the training data in mini-batches.
4. Compute the gradient of the objective function with respect to the parameters.
5. Update the moving average of squared gradients and model parameters using the update rule.
6. Repeat steps 3-5 until convergence or a predefined number of iterations.

## Benefits
- **Stabilized learning rates**: RMSprop adapts the learning rates based on the magnitude of recent gradients, effectively stabilizing the training process and preventing exploding gradients.
- **Improved convergence**: By scaling the learning rates appropriately, RMSprop can achieve faster convergence and better performance compared to standard optimization algorithms.
- **Applicability to various architectures**: RMSprop is widely used in training deep neural networks and can be applied to various architectures and types of data.

## Example
```python
# Pseudocode for RMSprop
E_g_squared = 0
for each iteration t:
    gradient = compute_gradient(data, parameters)
    E_g_squared = beta * E_g_squared + (1 - beta) * gradient^2
    parameters = parameters - (alpha / sqrt(E_g_squared + epsilon)) * gradient


# Adam (Adaptive Moment Estimation)

## Overview
Adam (Adaptive Moment Estimation) is an optimization algorithm commonly used in training neural networks and other machine learning models. It combines the benefits of both AdaGrad and RMSprop by maintaining separate adaptive learning rates for each parameter and incorporating momentum to speed up convergence.

## Algorithm
Adam computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients. It maintains exponentially decaying moving averages of both the gradients (first moments) and the squared gradients (second moments). The update rule for parameter θ is as follows:

m_t = β1 * m_{t-1} + (1 - β1) * g_t
v_t = β2 * v_{t-1} + (1 - β2) * g_t^2

m̂_t = m_t / (1 - β1^t)
v̂_t = v_t / (1 - β2^t)

θ = θ - (α / (sqrt(v̂_t) + ε)) * m̂_t

Where:
- g_t is the gradient of the objective function J(θ) with respect to the parameters θ at iteration t.
- m_t and v_t are moving averages of the gradients (first moments) and squared gradients (second moments), respectively.
- β1 and β2 are the decay rates for the first and second moments, typically set close to 1 (e.g., 0.9 and 0.999, respectively).
- α is the learning rate.
- ε is a small constant (usually on the order of 1e-8) added for numerical stability.

Adam combines the benefits of AdaGrad's adaptive learning rates and RMSprop's stability, while also incorporating momentum to speed up convergence. It automatically adapts the learning rates for different parameters and efficiently handles sparse gradients.

## Usage
1. Choose suitable values for the decay rates β1 and β2, the initial learning rate α, and the small constant ε.
2. Initialize the first and second moment estimates m_0 and v_0 to zero.
3. Iterate through the training data in mini-batches.
4. Compute the gradient of the objective function with respect to the parameters.
5. Update the first and second moment estimates and model parameters using the update rule.
6. Repeat steps 3-5 until convergence or a predefined number of iterations.

## Benefits
- **Adaptive learning rates**: Adam adapts the learning rates for different parameters based on estimates of first and second moments of the gradients, leading to faster convergence and improved performance.
- **Efficient handling of sparse gradients**: Adam efficiently handles sparse gradients by automatically adjusting the learning rates based on the magnitude of the gradients.
- **Combines momentum and adaptive learning rates**: Adam combines the benefits of momentum and adaptive learning rates, leading to faster convergence and better performance compared to other optimization algorithms.

## Example
```python
# Pseudocode for Adam
m = 0
v = 0
for each iteration t:
    gradient = compute_gradient(data, parameters)
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * gradient^2
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    parameters = parameters - (alpha / (sqrt(v_hat) + epsilon)) * m_hat
