
from classes.neuron import Neuron
from helpers.computePE import computePE
from helpers.activation import sigmoid, sigmoid_derivative

import json

def predict():
    with open("data/or_model.json", "r") as f:
        model_data = json.load(f)

    weights = model_data["weights"]
    bias = model_data["biases"]

    neuron = Neuron(2)
    neuron.update_weights(weights)
    neuron.bias = bias

    print("OR Gate using Neural Network")

    while True:
        input1 = float(input("Input 1: "))
        input2 = float(input("Input 2: "))

        if input1 not in [1.0, 0.0] or input2 not in [1.0, 0.0]:
            print("Only 0 and 1 is allowed!")
            continue
        break

    prediction = computePE([input1, input2], neuron)
    prediction = sigmoid(prediction)
    print(f"\nPrediction: {prediction}")
    if prediction < 0.5: answer = 0
    else: answer = 1

    print(f"Answer: {int(input1)} | {int(input2)} -> {answer}")



if __name__ == "__main__":
    predict()
