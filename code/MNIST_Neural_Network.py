
from __future__ import absolute_import
from matplotlib import pyplot as plt
import numpy as np
from preprocess import get_data


class Model:
    """
    This Model class contains the architecture of a single layer Neural Network. This model is used 
    for classifying MNIST with batched learning. This whole project was built from scratch using 
    only NumPy and Python built-in functions.
    """


    def __init__(self):
        # Initialize all hyperparametrs
        self.input_size = 784 # Size of image vectors
        self.num_classes = 10 # Number of possible classes
        self.batch_size = 100 
        self.learning_rate = 0.5

        # Initialize weights and biases
        self.W = np.zeros((self.input_size, self.num_classes))
        self.b = np.zeros((1, self.num_classes))


    def call(self, inputs):
        """
        This function builds the model. It forward pass on one batch of input images.

        Args:
            inputs (2D matrix): A normalized (0 to 1) input images matrix with size = batch_size*784

        Returns:
            [2D matrix]: Probability matrix with size = batch_size x 10
        """
        
        # Linear equation for the input
        L = inputs @ self.W + self.b
        # Raise to e to the L for softmax function
        exp_L = np.exp(L)
        # Calculate sum for probability
        sum_L = np.sum(exp_L, 1)
        # Reshape to 2D matrix for matrix division
        sum_L = sum_L.reshape(sum_L.shape[0], 1)
        # Calculate the probability matrix
        probabilities = exp_L / sum_L
        return probabilities
    

    def loss(self, probabilities, labels):
        """
        This is the model's loss function. It calculates the model cross-entropy loss after one 
        forward pass.

        Args:
            probabilities ([2D matrix]): This is the probabilites matrix we calculated in the call 
            function.

            labels ([Array]): This is the correct labels we passed in.

        Returns:
            [float]: Returns the average cross entropy loss for a batch
        """

        # Calculate the average loss for one batch of training examples.
        loss = -np.log(probabilities[:, labels])
        average = np.mean(loss)
        return average


    def back_propagation(self, inputs, probabilities, labels):
        """
        This function returns the gradients for the weights and bias of the model after one forward 
        pass and loss calculation.

        Args:
            inputs ([2D matrix]): Input training image matrix.
            probabilities ([2D matrix]): probabilities matrix calculated from call function. 
            labels ([array]): label array with correct answer.

        Returns:
            [2D matrix]: gradient for weights and bias.
        """

        # Create the one-hot vector y
        y = np.eye(self.batch_size)
        # Organize the vector with correct labels
        y = y[labels, 0:self.num_classes]
        # Use the gradient desent update rule
        gradW = self.learning_rate * (inputs.T @ (y - probabilities)) / self.batch_size
        gradB = self.learning_rate * np.sum(y - probabilities) / self.batch_size
        return gradW, gradB
    

    def accuracy(self, probabilities, labels):
        """
        Calculates the accuracy of the model. Achieved by comparing number of correct predictions
        with the correct answers.

        Args:
            probabilities ([2D matrix]): probabilities matrix calculated from call function. 
            labels ([array]): label array with correct answer.

        Returns:
            [float]: Accuracy of the model.
        """

        # Get the index for output in probabilities matrix.
        prediction = np.argmax(probabilities, axis=1)
        total_correct = sum(prediction == labels)
        # Calculate the accuracy of the model.
        accuracy = total_correct / labels.shape[0]
        return accuracy


    def gradient_descent(self, gradW, gradB):
        """
        Use gradient descent on the model.

        Args:
            gradW ([2D matrix]): gradient for weights
            gradB ([1D matrix]): gradient for biases
        """

        # Update weight and bias
        self.W += gradW
        self.b += gradB
    



def train(model, train_inputs, train_labels):
    """
    Train the model using given inputs and labels

    Args:
        model ([Model object]): Initialized model for forward pass and backward propagtion.
        train_inputs ([2D matrix]): Training inputs (All images)
        train_labels ([array]): Training labels (All labels)
    """
    # Create a loss array to check our loss for each batch
    loss = np.empty(int(train_inputs.shape[0] / model.batch_size))
    # Iterate over the training inputs and labels, in model.batch_size increments
    for i in range(int(train_inputs.shape[0] / model.batch_size)):
        # Get batch size input and labels
        inputs = train_inputs[model.batch_size*i : model.batch_size*(i+1), :]
        labels = train_labels[model.batch_size*i : model.batch_size*(i + 1)]
        probabilities = model.call(inputs)
        # Compute the gradient and bias for every batch.
        gradW, gradB = model.back_propagation(inputs, probabilities, labels)
        # Update the model using gradient descent.
        model.gradient_descent(gradW, gradB)
        # Add loss of this batch to the array
        np.append(loss, model.loss(probabilities, labels))



def test(model, test_inputs, test_labels):
    """
    Test the model using test inputs and test labels.

    Args:
        model ([Model object]): Our trained model.
        test_inputs ([2D matrix]): MNIST test inputs.
        test_labels ([array]): MNIST test labels.

    Returns:
        [float]: accuracy of the model.
    """

    # Use the call function in the model to get the probability matrix.
    probabilities = model.call(test_inputs)
    # Calculate the accuracy using the accuracy function.
    accuracy = model.accuracy(probabilities, test_labels)
    return accuracy


def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the results of our model.

    Args:
        image_inputs ([images]): image data from get_data()
        probabilities ([2D matrix]): probabilities matrix from model.call()
        image_labels ([array]): correct label data
    """

    images = np.reshape(image_inputs, (-1, 28, 28))
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()




def main():
    """
    Read in MNIST data and initialize, train and test the model.
    """

    # Read in MNIST train and test examples.
    train_inputs, train_labels = get_data("C:/data/train-images-idx3-ubyte.gz",
                                   "C:/data/train-labels-idx1-ubyte.gz", 60000)
    test_inputs, test_labels = get_data("C:/data/t10k-images-idx3-ubyte.gz",
                                 "C:/data/t10k-labels-idx1-ubyte.gz", 10000)
    # Create Model
    model = Model()
    # Train model using training inputs and labels.
    train(model, train_inputs, train_labels)
    # Print the accuracy of the model.
    print("Training accuracy is: ", test(model, test_inputs, test_labels))
    # Visualize the data using visualize_results()
    inputs = test_inputs[0:10, :]
    labels = test_labels[0:10]
    probabilities = model.call(inputs)
    visualize_results(inputs, probabilities, labels)
    
if __name__ == '__main__':
    main()
