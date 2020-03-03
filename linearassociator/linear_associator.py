#Shuvrima Alam
import numpy as np
import math
class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        self.input_dimensions = input_dimensions
        self.number_of_nodes = number_of_nodes
        self.transfer_function = transfer_function        
        self.initialize_weights()

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        self.num_neurons = self.number_of_nodes
        if seed != None:
            np.random.seed(seed)
            self.weights = np.random.randn(self.num_neurons, self.input_dimensions)
        else:
            self.weights = np.random.randn(self.num_neurons, self.input_dimensions)        
    def set_weights(self, W):
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
         """
        if W.shape == (self.num_neurons, self.input_dimensions):
            self.weights=W
            return None
        else:
            return -1

    def get_weights(self):
        """
         This function should return the weight matrix(Bias is included in the weight matrix).
         :return: Weight matrix
         """
        return self.weights
    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        self.Y_Predict = np.zeros([self.number_of_nodes, X[0].size])
        #transpose so that we can iterate with input array
        self.Y_Predict = self.Y_Predict.transpose()
        self.X_train = X
        # '''predict using hardlimit'''
        #transpose the input array for iterating
        self.X_train = self.X_train.transpose()

        #iterating over the samples
        for (input_sample, output) in zip(self.X_train, self.Y_Predict):
            #iterate over the predictions made by each neuron
            for neuron in range(self.num_neurons):
                prediction = None
                net = 0
                #iterate over the inputs in a sample and the corresponsing weights
                for (input, weight) in zip(input_sample, self.weights[neuron]):
                    #linear combination of weights and inputs
                    net = net + (input * weight)
                if self.transfer_function == 'Hard_limit':
                    prediction = self.hard_limit(net)
                elif self.transfer_function == 'Linear':
                    prediction = self.linear(net)
                #update predict output array
                output[neuron] = prediction
        #returing in the same shape as professor has given inputs in
        return(self.Y_Predict.transpose())
    def hard_limit(self, net):
        if net >= 0:
            prediction = 1
            return(prediction)
        else:
            prediction = 0
            return(prediction)
    def linear(self, net):
        prediction = net
        return(prediction)            
    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        self.Xtrain=X
        self.Ytrain=y
        Xplus=np.linalg.pinv(self.Xtrain)
        self.weights= y @ Xplus
    def Delta(self, w, alpha, err, sample):
        w = np.add(w, ((((sample.transpose()).dot(err)).transpose()).dot(alpha)))
        return(w)

    def Filtered(self, w, gamma, alpha, target, sample):
        w = np.add((w.dot((1 - gamma))), ((((sample.transpose()).dot(target)).transpose()).dot(alpha)))
        return(w)

    def Unsupervised_Hebbian(self, w, alpha, actual, sample):
        w = np.add(w, ((((sample.transpose()).dot(actual)).transpose()).dot(alpha)))
        return(w)
    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        for epoch in range(num_epochs):
            input = X
            self.Y_Predict = self.predict(input).transpose() #transpose for ease of iteration
            self.Y_train = y.transpose()
            self.X_train = X
            self.X_train = self.X_train.transpose()
            batches = math.ceil(len(self.X_train)/batch_size) #no of batches
            batch_lower_index = 0
            batch_upper_index = batch_size
            #iterate through the batches
            for i in range(batches):
                input_batch = self.X_train[batch_lower_index : batch_upper_index]
                target_batch = self.Y_train[batch_lower_index : batch_upper_index]
                actual_batch = self.Y_Predict[batch_lower_index : batch_upper_index]
                error = np.subtract(target_batch, actual_batch)
                if learning == 'delta' or 'Delta':
                    self.weights = self.Delta(self.weights, alpha, error, input_batch)
                elif learning == 'Filtered' or 'filtered':
                    self.weights = self.Filtered(self.weights, gamma, alpha, target_batch, input_batch)
                elif learning == 'Unsupervised_hebb' or 'unsupervised_hebb':
                    self.weights = self.Unsupervised_Hebbian(self.weights, alpha, actual_batch, input_batch)
                #Updating the actual outputs batch-wise
                input = X
                self.Y_Predict = self.predict(input).transpose()
                batch_lower_index = batch_lower_index + batch_size
                batch_upper_index = batch_upper_index + batch_size

    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        self.Y_train = y #transpose()
        input = X
        self.Y_Predict = self.predict(input) #transpose()
        mse = np.mean((self.Y_train - self.Y_Predict)**2)

        return(mse)
     
