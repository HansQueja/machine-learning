
def computePE(inputs, neuron):
    neuron_value = 0
    for i in range(len(inputs)):
        neuron_value += (inputs[i] * neuron.weights[i])
    return neuron_value + neuron.bias