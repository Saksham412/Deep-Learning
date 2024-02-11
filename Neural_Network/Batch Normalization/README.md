# Batch Normalization 

[Paper](https://arxiv.org/abs/1502.03167)

Batch Normalization is a technique used in deep learning to improve the training of artificial neural networks. It aims to address the problem of internal covariate shift, which refers to the change in the distribution of network activations due to changes in the parameters during training.

Here's how Batch Normalization works:

- Normalization: In each training mini-batch, Batch Normalization normalizes the activations of each layer by subtracting the mean and dividing by the standard deviation of the batch.

- Scaling and Shifting: After normalization, the activations are scaled and shifted using learnable parameters (gamma and beta). This allows the model to learn the optimal scale and shift for each feature.

- Stabilization: Batch Normalization helps stabilize the training process by reducing the internal covariate shift. This enables the use of higher learning rates, which can speed up the training process.

- Regularization: Batch Normalization acts as a form of regularization by adding noise to the activations through the normalization process, similar to dropout. This can help prevent overfitting and improve the generalization of the model.

```
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

# Create a Sequential model
model = Sequential()

# Add the input layer and the first hidden layer with Batch Normalization
model.add(Dense(64, input_shape=(input_dim,), activation='relu'))
model.add(BatchNormalization())

# Add more hidden layers with Batch Normalization
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())

# Add the output layer
model.add(Dense(output_dim, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()
```