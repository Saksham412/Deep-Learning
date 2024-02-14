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
