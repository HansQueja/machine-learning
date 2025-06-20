"""
This will be the working neuron class for all neural network side projects
"""

from math import sqrt
from numpy.random import randn, uniform, randint, normal

class Neuron:
    def __init__(self, no_weights, init = "kaiming"):
        self.weights = [0] * no_weights
        self.bias = uniform(0, 0.1)
        self.total_weights = no_weights
        self.initialize_weights(init, no_weights)

    def __str__(self):
        rstring = f"Number of weights: {self.total_weights}\n"
        for i, weight in enumerate(self.weights):
            rstring += f"Weight #{i}: {weight}\n"
        rstring += f"Bias: {self.bias}"
        return rstring

    def initialize_weights(self, init, no_weights):
        if init == "kaiming":
            weights = self.kaiming_init()
            for i in range(no_weights):
                self.weights[i] = weights[i]
        else:
            for i in range(no_weights):
                self.weights[i] = self.xavier_init()

    def update_weights(self, new_weights):
        for i in range(self.total_weights):
            self.weights[i] = new_weights[i]

    # Used for relu
    def kaiming_init(self):
        std = sqrt(2.0 / self.total_weights)
        return normal(0.0, std, (1, self.total_weights)).tolist()[0]
    
    # Used for sigmoid and tan
    def xavier_init(self):
        lower, upper = -(1.0 / sqrt(self.total_weights)), (1.0 / sqrt(self.total_weights))
        return uniform(lower, upper)
 