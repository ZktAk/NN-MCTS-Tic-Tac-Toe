"""
This code was taken from: https://github.com/miloharper/multi-layer-neural-network
And adapted for use in this application.
"""

import numpy
from numpy import exp, array, random, dot
import matplotlib.pyplot as plt


# The Sigmoid function, which describes an S shaped curve.
# We pass the weighted sum of the inputs through this function to
# normalise them between 0 and 1.
def _sigmoid(x):
    return 1 / (1 + exp(-x))


# The derivative of the Sigmoid function.
# This is the gradient of the Sigmoid curve.
# It indicates how confident we are about the existing weight.
def _sigmoid_derivative(x):
    return x * (1 - x)


class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        # A matrix of size number_of_neurons x number_of_inputs_per_neuron
        # filled with elements in the half-open interval [-1,1)
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, input_size, hidden_size, hidden_layers, output_size):

        self.layers = []

        # ----------------------------------
        # Initialize fist hidden layer
        # ----------------------------------
        self.layers.append(NeuronLayer(hidden_size, input_size))

        # ----------------------------------
        # Initialise rest of hidden layers
        # ----------------------------------
        for n in range(hidden_layers - 1):
            self.layers.append(NeuronLayer(hidden_size, hidden_size))

        # ----------------------------------
        # Initialize output layer
        # ----------------------------------
        self.layers.append(NeuronLayer(output_size, hidden_size))

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pass the training set through our neural network
            output_from_layers = self.predict(input_vectors)

            deltas = []

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            lastLayer_error = targets - output_from_layers[-1]
            lastLayer_delta = lastLayer_error * _sigmoid_derivative(output_from_layers[-1])
            deltas.append(lastLayer_delta)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            for n in range(len(output_from_layers)-1):
                layer_error = deltas[-1].dot(self.layers[-1-n].synaptic_weights.T)
                layer_delta = layer_error * _sigmoid_derivative(output_from_layers[-1-(n+1)])
                deltas.append(layer_delta)

            # Calculate how much to adjust the weights by
            layer1_adjustment = input_vectors.T.dot(deltas[-1])
            layer_adjustments = [layer1_adjustment]

            for n in range(len(deltas)-1):
                layer_adjustments.append(output_from_layers[n].T.dot(deltas[-2-n]))

            # Adjust the weights.
            for n in range(len(self.layers)):
                self.layers[n].synaptic_weights += layer_adjustments[n]

            # save pertinent data for graph
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)[-1]
                    error = numpy.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors

    # The neural network thinks.
    def predict(self, inputs):

        layer_output = inputs
        outputs = []

        for layer in self.layers:
            layer_output = _sigmoid(dot(layer_output, layer.synaptic_weights))
            outputs.append(layer_output)
        return outputs

    # The neural network prints its weights
    def print_weights(self):
        print("    Layer 1 (4 neurons, each with 3 inputs): ")
        print(self.layers[0].synaptic_weights)
        print("    Layer 2 (4 neurons, each with 4 inputs):")
        print(self.layers[1].synaptic_weights)
        try:
            print("    Layer 3 (1 neuron, with 4 inputs):")
            print(self.layers[2].synaptic_weights)
        except:
            pass

if __name__ == "__main__":

    #Seed the random number generator
    random.seed(1)

    # Create a neural network
    neural_network = NeuralNetwork(3, 4, 2, 1)

    print("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    training_input_vectors = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    training_targets = array([[0, 1, 1, 1, 1, 0, 0]]).T

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    training_error = neural_network.train(training_input_vectors, training_targets, 60000)

    print("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()

    # Test the neural network with a new situation.
    print("Stage 3) Considering a new situation [1, 1, 0] -> ?: ")
    Layer_outputs = neural_network.predict(array([1, 1, 0]))
    print(Layer_outputs[-1])

    plt.plot(training_error)
    plt.xlabel("Iterations (100s)")
    plt.ylabel("Error for all training instances")
    plt.savefig("cumulative_error.png")

    print(f"\nError at End {training_error[len(training_error) - 1]}")
