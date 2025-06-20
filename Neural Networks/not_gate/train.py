
from helpers.parser import excel_parser
from helpers.activation import sigmoid, sigmoid_derivative
from helpers.computePE import computePE
from classes.neuron import Neuron

import json

def train():
    training_data = excel_parser("not_training.xlsx")
    
    mse = 1
    min_error = 1e-5

    epoch = 0
    max_epoch = 100000

    learning_rate = 0.05

    neuron = Neuron(1)
    while epoch < max_epoch and mse > min_error:
        errors = []
        for index, row in training_data.iterrows():
            input1 = row['Input A']
            target = row['Output']

            neuron_value = computePE([input1], neuron)
            hypothesis = sigmoid(neuron_value)
            print(hypothesis)
            diff = hypothesis - target
            cur_error = pow(diff, 2) / 2
            errors.append(cur_error)
            
            grad = diff * sigmoid_derivative(neuron_value)

            neuron.weights[0] -= learning_rate * grad * input1
            neuron.bias -= learning_rate * grad * 1
        
        mse = sum(errors) / len(errors) 
        if mse < min_error:
            print(f"Converged to {min_error}")
            break
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}      MSE: {mse: .8f}")
        epoch += 1

    model_data = {
        "weights": neuron.weights,
        "biases": neuron.bias
    }

    with open("data/not_model.json", "w") as f:
        json.dump(model_data, f)
        
    print(f"MSE: {mse}\nTraining Complete.")


if __name__ == "__main__":
    train()