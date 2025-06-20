
from classes.neuron import Neuron
from helpers.computePE import computePE
from helpers.activation import sigmoid, sigmoid_derivative

import json

def predict():
    with open("data/not_model.json", "r") as f:
        model_data = json.load(f)

    weights = model_data["weights"]
    bias = model_data["biases"]

    neuron = Neuron(1)
    neuron.update_weights(weights)
    neuron.bias = bias

    print("NOT Gate using Neural Network")

    while True:
        user_input = float(input("Provide a binary input: "))

        if user_input not in [1.0, 0.0]:
            print("Only 0 and 1 is allowed!")
            continue
        break

    prediction = computePE([user_input], neuron)
    prediction = sigmoid(prediction)
    print(f"\nPrediction: {prediction}")
    if prediction < 0.5: answer = 0
    else: answer = 1

    print(f"Answer: {int(user_input)} -> {answer}")



if __name__ == "__main__":
    predict()
