# Deep Learning Notes

## Introduction 

### 1. Deep Learning vs Machine Learning

```
|                            Deep Learning                                      |                    Machine Learning                                     

| Works primarly on Neural Networks                                             |   Works on Stastical data and interpretation                

| Works Well with Large Data                                                    |   Works well  with smaller data

| ANN(Basic), RNN(Text and speech), CNN(Image), GAN(Generative data)            |   Regression and Classification with stastical tools and models

| Extracts relevant features progressively                                      |   Features have to be provided|(Can think with and example)

| Takes more time to train                                                      |   Takes less time to train

| Need more hardware power for computation                                      |   Releatively easier to compute

| Not intrepretable as it extracts features on its own we cant give the reason  |   Interpretable as majorly stastics is used

 ```

### 2. Types of Neural networks

    1. ANN-> Artificial Neural Network is the basic architecture of the neural network
    2. RNN-> Recuurent Neral Networks are used in NLP. Here the perceptrons gives feedback to itself.
    3. CNN-> Convolutional Neural networks are used in image processing.
    4. Auto Encoded-> 
    5. GAN-> Generative adverserial networks can imagine things on ther own with generator discriminator pair.


### 3. History

1. 60's - Perceptron Invention and Initial Boom:**
   - In the 1960s, the perceptron, a basic building block of neural networks, was invented.
   - Perceptrons could only handle linear functions and struggled with non-linear functions, notably failing to learn the XOR function.
   - Due to these limitations, interest in neural networks diminished during this period.

2. 80's - Stacking Perceptrons and Challenges:**
   - In the 1980s, researchers discovered that stacking perceptrons into layers (creating neural networks) could overcome the limitations of handling non-linear functions.
   - However, during this time, there wasn't enough data available for training, and the weights of the networks were assigned randomly.
   - Other algorithms such as random forests and support vector machines (SVM) outperformed neural networks in certain tasks, leading to a decline in interest in deep learning.

3. 2006 Onwards - Resurgence of Deep Learning:**
   - Starting around 2006, there was a resurgence of interest in deep learning.
   - One crucial development was the initialization of weights using pre-trained unsupervised networks, such as autoencoders.
   - The pivotal moment came in 2010 and 2011 with the ImageNet competition for image classification. Neural networks, especially convolutional neural networks (CNNs), outperformed traditional machine learning algorithms.
   - This success demonstrated the effectiveness of deep learning, leading to its widespread adoption in various fields, including computer vision, natural language processing, and more.
   - Since then, deep learning has continued to evolve, with improvements in architectures, training techniques, and the availability of large datasets, establishing itself as a dominant paradigm in machine learning.


### 4. Application areas

1. Self driving cars
2. Game playing agetns
3. Virtiual assistants
4. Image colorization
5. Image caption generatino
6. Music generation 
7. Unbluring the images and many more.

## Perceptron

A perceptron is the simplest form of a neural network, serving as a basic building block for more complex models. It's often used for binary classification tasks.

![Perceptron](images/perceptron.png)

### Comparison with Neuron

- **Complexity**: A single neuron in a biological brain is highly complex, involving various biological processes. In contrast, a perceptron is a simplified mathematical model, consisting of input features, weights, and an activation function.
  
- **Processing**: Neurons in the brain process information in parallel, forming intricate networks. Perceptrons, on the other hand, process inputs sequentially, updating weights based on input signals and predefined rules.
  
- **Neuroplasticity**: Biological neurons exhibit neuroplasticity, allowing them to adapt and reorganize connections based on experience. Perceptrons lack this capability and require manual adjustments to their parameters for learning.

Weight values in a perceptron indicate the importance of features in the classification process.

### Geometric Intuition

The perceptron classifies data into two classes by creating a decision boundary, which could be a line in 2D, a plane in 3D, or a hyperplane in higher dimensions. This decision boundary separates the classes based on the features of the input data.
