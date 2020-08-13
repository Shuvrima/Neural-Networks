# Alam, Shuvrima
# 1001-085-726
# 2020_03_23
# Assignment-03-01
# %tensorflow_version 2.x
# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np
import math

class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.input_dimension=input_dimension
        self.weights = []
        self.biases = []
        self.activations = []
        self.loss = None       
    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """
        self.weights.append(tf.zeros(shape = (self.input_dimension, num_nodes), dtype = 'float64'))
        self.input_dimension = num_nodes
        self.biases.append(tf.zeros(shape = ( 1, num_nodes), dtype = 'float64'))
        if transfer_function == "Relu":
            transfer_function =self.relu
            self.activations.append(transfer_function)
        elif transfer_function == "Sigmoid":
            transfer_function =self.sigmoid
            self.activations.append(transfer_function)
        elif transfer_function == "Linear":
            transfer_function=self.linear
            self.activations.append(transfer_function)   
    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
        return(self.weights[layer_number])
    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
        return(self.biases[layer_number])
    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        self.weights[layer_number] = weights
    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.biases[layer_number] = biases
    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))        
    def sigmoid(self, x):

        return tf.nn.sigmoid(x)

    def linear(self, x):
        return x

    def relu(self, x):
        return tf.nn.relu(x)
    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        self.X_train = tf.Variable(X)
        #Going over the layers
        for (w, b, activation_function) in zip(self.weights, self.biases, self.activations):
            net = tf.matmul(self.X_train, w) + b
            self.predictY = activation_function(net)
            if w is self.weights[-1]:
                return(self.predictY)
            else:
                self.X_train = self.predictY
                continue
    def encodify(self, y):  #Using one hot encodify
        
        onehot_encoded_val = list()
        for value in y:
            out = [0 for _ in range(self.weights[-1].shape[1])]
            out[value] = 1
            onehot_encoded_val.append(out)
        onehot_encoded_val = np.array(onehot_encoded_val)
        return(onehot_encoded_val)
    
    def batch_training(self, x, y, learning_rate):  #batch processing for training
        with tf.GradientTape() as tape:
            predictions = self.predict( x)
            loss = self.loss(y, predictions)
            new_w, new_b = tape.gradient(loss, [self.weights, self.biases])
        for ((ind, w), b) in zip(enumerate(self.weights), self.biases):
            w.assign_sub(learning_rate * new_w[ind])
            b.assign_sub(learning_rate * new_b[ind])
        return None            

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """
        self.Y_train = tf.Variable(y_train)
        self.X_train = tf.Variable(X_train)

        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.batch(batch_size)

        for epoch in range(num_epochs): #call batch wise training and go over layers
            for step, (x, y) in enumerate(dataset):
                self.batch_training(x, y, alpha)
    def set_loss_function(self, total_loss):
        self.loss = total_loss #for loss function
    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        self.X_train = tf.Variable(X)
        self.Y_train = self.encodify(y)
        self.predictY = self.predict(X)

        Ytrain = tf.Variable(self.Y_train).numpy()
        Ytrain = Ytrain.tolist()
        predictY = tf.Variable(self.predictY).numpy()
        predictY = predictY.tolist()

        num_samples = 0
        err_count = 0 #counting error
        for (input_sample, t_val, a_val) in zip(self.X_train, Ytrain, predictY):
            num_samples = num_samples + 1
            if np.argmax(t_val) != np.argmax(a_val):
                err_count = err_count + 1
        percent_error = (err_count/num_samples)
        return(percent_error)


    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        self.X_test = tf.Variable(X)
        self.Y_test = self.encodify(y)
        self.predictY = self.predict(X)
        confusion_matrix = np.zeros(shape = (self.weights[-1].shape[1], self.weights[-1].shape[1]))

        for a, b in zip(self.Y_test, self.predictY):
            i = np.argmax(a)
            j = np.argmax(b)
            confusion_matrix[i][j] += 1
        return(confusion_matrix)
