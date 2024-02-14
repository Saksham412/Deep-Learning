- **Challanges:** 
1. Learning rate initialsation
2. Learning rate scheduling
3. separte learning rate for each dimension
4. Local minima
5. Saddle point

## Exponentially weighted moving average(EWMA)

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

