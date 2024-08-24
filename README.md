# DIP_2022_1


### Bonus Assignment: MNIST

This bonus assignment is optional and will be graded with extra points awarded to a very small number of top-performing students:


### Task Overview

For those interested in attempting the bonus task, follow these steps:

1. **Increase Dataset Size**:
   - Expand the training and testing datasets as follows:
     ```python
     mnist_train = torch.utils.data.Subset(mnist_train, range(0, 6000))
     mnist_test  = torch.utils.data.Subset(mnist_test , range(0, 1000))
     ```

2. **Modify Neural Network Structure**:
   - Adjust the neural network architecture and fine-tune hyperparameters to maximize test accuracy.
   - Document your changes and report the final test accuracy in your submission. Please note that I will verify the results myself.

### Guidelines

- **Unchangeable Parameters**:
   - **Number of Epochs**: Fixed at 20 (increased epochs could give an unfair advantage).
   - **Dataset Size**: Fixed at 6000 for training and 1000 for testing.

- **Project Structure**:
   - Do not modify the dataset directory structure. Keep it as:
     ```python
     root="./"
     ```

- **Network Type**:
   - Use a fully connected neural network (FCNN), not a CNN.

- **Allowed Modifications**:
   - Number of layers
   - Number of nodes per layer
   - Activation functions
   - Learning rate
   - Alternative optimizers (other than SGD)
   - Techniques like batch normalization, dropout, regularization, and data augmentation
   - For any other methods (e.g., weight normalization, layer normalization, LR scheduler), please consult with me before using them.


------------------------------------------------------------------------------------


### Results

The results from the bonus assignment are summarized below:

1. **Test Accuracy**:
   - The highest test accuracy achieved was **95.10%**.

2. **Key Modifications**:
   - **Hyperparameter** : An optimal learning rate of **0.002**, minibatch size 64 was identified.
   - **Optimizer**: Switching to **Adam** improved accuracy.
   - **Weight Initialization**: Using **Kaiming He** improved accuracy by **1%**. (appropriate function for activation function ReLU)
   - **Network Structure**: The final model used 6 layers with nodes decreasing by 2^n.

4. **Final Model**:
   - **Layers**: 6
   - **Nodes per Layer**: (784, 512, 256, 128, 64, 32)
   - **Optimizer**: Adam with a learning rate of **0.02**
   - **Weight Initialization**: Kaiming He
  
5. **Additional Technique Trials**
   - **Batch Normalization** & **Dropout** : suppressed overfitting, but decreased the final test accuracy as well.
   - **Optimization Function** : Using Sigmoid works better with less layers, but didn't give the best result.

6. **Summary**:
   - These modifications led to a final test accuracy of **95.10%**, improving overall model performance.
