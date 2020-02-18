# Alam, Shuvrima
import numpy as np

class Perceptron(object):
    def __init__(self, input_dimensions=2,number_of_nodes=4):
        """
        Initialize Perceptron model and set all the weights and biases to random numbers.
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: Note that number of neurons in the model is equal to the number of classes.
        """
        self.input_dimensions = input_dimensions
        self.number_of_nodes = number_of_nodes
        self.initialize_weights()
    def initialize_weights(self,seed=None):
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
            self.weights = np.random.randn(self.num_neurons, self.input_dimensions + 1)
        else:
            self.weights = np.random.randn(self.num_neurons, self.input_dimensions + 1)
            
    def set_weights(self, W):
        """
        This function sets the weight matrix (Bias is included in the weight matrix).
        :param W: weight matrix
        :return: None if the input matrix, w, has the correct shape.
        If the weight matrix does not have the correct shape, this function
        should not change the weight matrix and it should return -1.
        """
        if W.shape == (self.num_neurons, self.input_dimensions + 1):
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
        Make a prediction on a batach of inputs.
        :param X: Array of input [input_dimensions,n_samples]
        :return: Array of model [number_of_nodes ,n_samples]
        Note that the activation function of all the nodes is hard limit.
        """
        self.predictY = np.zeros([self.number_of_nodes, X[0].size], int)
        self.predictY = self.predictY.transpose()
        self.X_train = np.concatenate((np.ones([1,X[0].size]),X))
        self.X_train = self.X_train.transpose()
        for (input_sample, output) in zip(self.X_train, self.predictY):
            for neuron in range(self.num_neurons):
                res = None
                outcome = 0
                for (input, weight) in zip(input_sample, self.weights[neuron]):
                    outcome = outcome + (input * weight)
                if outcome >= 0:
                    res = 1
                else:
                    res = 0
                output[neuron] = res
        return(self.predictY.transpose())

    def train(self, X, Y, num_epochs=10,  alpha=0.1):
        """
        Given a batch of input and desired outputs, and the necessary hyperparameters (num_epochs and alpha),
        this function adjusts the weights using Perceptron learning rule.
        Training should be repeated num_epochs times.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """
        for epoch in range(num_epochs):
            input = X
            self.predictY = self.predict(input).transpose()
            self.Y_train = Y.transpose()
            self.X_train = np.concatenate((np.ones([1,X[0].size]),X)) ##adding 1
            self.X_train = self.X_train.transpose() #transpose for ease of iteration
            
            #Checking all samples
            for (input_sample, output2, output2_row) in zip(self.X_train, self.Y_train, range(len(self.Y_train))):
                output = self.predictY[output2_row]
                for neuron in range(self.num_neurons): #checking for each neuron
                    error = output2[neuron] - output[neuron]
                    for (input, weight, weight_index) in zip(input_sample, self.weights[neuron], range(len(self.weights[neuron]))):
                        self.weights[neuron][weight_index] = weight + (alpha * error * input)
                input = X
                self.predictY = self.predict(input).transpose()
    def calculate_percent_error(self,X, Y):
        """
        Given a batch of input and desired outputs, this function calculates percent error.
        For each input sample, if the output is not the same as the desired output, Y,
        then it is considered one error. Percent error is 100*(number_of_errors/ number_of_samples).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :return percent_error
        """ 
        self.X_train = np.concatenate((np.ones([1,X[0].size]),X))
        self.X_train = self.X_train.transpose()
        self.Y_train = Y.transpose()
        input = X
        self.predictY = self.predict(input).transpose()
        num_samples = 0
        errcount = 0
        for (input_sample, output2, output) in zip(self.X_train, self.Y_train, self.predictY):
            num_samples = num_samples + 1
            for neuron in range(self.num_neurons):
                if output2[neuron] != output[neuron]:
                    errcount = errcount + 1
                    break
        percent_error = (errcount/num_samples)*100
        return(percent_error)
        
if __name__ == "__main__":
    input_dimensions = 2
    number_of_nodes = 2

    model = Perceptron(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes)
    model.initialize_weights(seed=2)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    print(model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    print("****** Model weights ******\n",model.get_weights())
    print("****** Input samples ******\n",X_train)
    print("****** Desired Output ******\n",Y_train)
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.1)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("******  Percent Error ******\n",percent_error)
    print("****** Model weights ******\n",model.get_weights())
