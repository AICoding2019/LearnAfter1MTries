import math
import json
import numpy as np
from random import random
from random import choice


class Neuron:
    def __init__(self):
        self.neuron = {
            'ID': [],
            'Weights': [],
            'Bias': [],
            'Enabled': 1,
            'Recurrent': [],
            'Layer': [],
            'ActivationFunction': [],
            'NodeNum': 0,
            'Inputs': [],
            'Output': 0,
        }

    def __str__(self):
        return '<%s>' % self.neuron

    @staticmethod
    def Sigmoid(u):
        return 1 / (1 + math.exp(-u))

    @staticmethod
    def Relu(u):
        return max(0, u)

    @staticmethod
    def LeakyRelu(u):
        return max(0.1 * u, u)

    @staticmethod
    def Tanh(u):
        return math.tanh(u)

    @staticmethod
    def Linear(u):
        return u

    def UpdateNeuron(self):
        inputs = np.array(self.neuron['Inputs'])
        weights = np.array(self.neuron['Weights'])
        diff = len(self.neuron['Inputs']) - len(self.neuron['Weights'])

        if diff > 0:
            weights = np.pad(weights, (0, diff), 'constant', constant_values=(0, 0))
        if diff < 0:
            inputs = np.pad(inputs, (0, -diff), 'constant', constant_values=(0, 0))

        bias = np.array(self.neuron['Bias'])
        Out = self.neuron['Output']
        recurrent = self.neuron['Recurrent']

        Output = inputs.dot(weights.transpose()) + bias + Out * recurrent
        # print('Output {}'.format(Output))

        if self.neuron['ActivationFunction'] == 'sigmoid':
            self.neuron['Output'] = self.Sigmoid(Output)

        if self.neuron['ActivationFunction'] == 'relu':
            self.neuron['Output'] = self.Relu(Output)

        if self.neuron['ActivationFunction'] == 'tanh':
            self.neuron['Output'] = self.Tanh(Output)

        if self.neuron['ActivationFunction'] == 'leakyRelu':
            self.neuron['Output'] = self.LeakyRelu(Output)

        if self.neuron['ActivationFunction'] == 'linear':
            self.neuron['Output'] = self.Linear(Output)

        # print('Act Output {}'.format(self.neuron['Output']))


class NeuralNetwork:
    def __init__(self, numInputs, numOutputs):
        self.numInputs = numInputs
        self.numOutputs = numOutputs

        self.Neuron = Neuron()

        self.Network = []
        self.activationFunc = ['sigmoid', 'relu', 'tanh', 'leakyRelu', 'linear']
        self.NeuralNet = {
            'Network': [],
            'Output': None,
            'Layers': 0,
            'Species': [-1, -1],  # [ num hiddenlayers, layers]
        }

    # self.CreateInitialGraph()

    def __str__(self):
        return '<%s>' % self.Network

    def CreateInitialGraph(self):
        # Each Input connected to each output neuron and it activation function is a sigmoid
        for outs in range(0, self.numOutputs):
            neuron = self.Neuron.neuron.copy()
            neuron['Layer'] = -1
            neuron['ActivationFunction'] = choice(self.activationFunc)
            neuron['ID'] = outs
            neuron['NodeNum'] = outs
            neuron['Weights'] = [random() for inputs in range(0, self.numInputs)]
            neuron['Bias'] = random()
            neuron['Recurrent'] = random()
            neuron['Enabled'] = choice([0, 1])
            neuron['Output'] = 0
            self.Network.append(neuron)
            neuron = []

        self.NeuralNet['Network'] = self.Network
        self.NeuralNet['Species'] = [0, self.numOutputs]

    def UpdateGraph(self, NetInputs):
        nextLayerInput = []
        nextnextLayerInput = []
        layerList = [neuron['Layer'] for neuron in self.Network]

        Iter = 0
        for index, neuron in enumerate(self.Network):
            if neuron['Layer'] == 0:
                neuron['Inputs'] = NetInputs
                self.Neuron.UpdateNeuron()
                neuron['Output'] = self.Neuron.neuron['Output']
                nextLayerInput.append(self.Neuron.neuron['Output'])
            else:
                if neuron['Layer'] == layerList[index]:
                    neuron['Inputs'] = nextLayerInput.copy()
                    self.Neuron.UpdateNeuron()
                    neuron['Output'] = self.Neuron.neuron['Output']
                    nextnextLayerInput.append(self.Neuron.neuron['Output'])
                    Iter += 1
                    if layerList.count(layerList[index]) == Iter:
                        Iter = 0
                        nextLayerInput = nextnextLayerInput.copy()
                        nextnextLayerInput = []

    @staticmethod
    def UserDefineGraph(numInputs, HiddenLayers, numOutputs, activationFunction='relu'):
        network = NeuralNetwork(numInputs, numOutputs)
        neuron = network.Neuron.neuron

        ID = 0
        for layer, nodes in enumerate(HiddenLayers):
            neuron['Layer'] = layer
            for node in range(0, nodes):
                neuron['ID'] = ID
                neuron['Weights'] = [random() for inputs in range(0, nodes)]
                neuron['Bias'] = random()
                neuron['ActivationFunction'] = activationFunction
                neuron['Recurrent'] = random()
                neuron['Enabled'] = 1
                neuron['Output'] = 0
                neuron['NodeNum'] = ID
                ID += 1
                network.Network.append(neuron.copy())

        for outs in range(0, numOutputs):
            neuron['Layer'] = -1
            neuron['ID'] = ID
            neuron['Weights'] = [random() for inputs in range(0, nodes)]
            neuron['Bias'] = random()
            neuron['ActivationFunction'] = activationFunction
            neuron['Recurrent'] = random()
            neuron['Enabled'] = 1
            neuron['Output'] = 0
            neuron['NodeNum'] = ID
            ID += 1
            network.Network.append(neuron.copy())

        network.NeuralNet['Network'] = network.Network
        network.NeuralNet['Layers'] = len(HiddenLayers) + 1
        network.NeuralNet['Species'] = [len(HiddenLayers), ID]
        return network

    def DrawGraph(self):
        pass

    def LogGraph(self, directory, filename):
        with open(directory  + filename + '.json', 'w') as json_file:
            json.dump(self.NeuralNet, json_file, indent=4)


if __name__ == '__main__':
    # Net = NeuralNetwork(2, 2)
    # print(Net.Network[0])
    # print(Net.Network[1])
    # print('-----')
    # Net.UpdateGraph([1, 1])
    # print(Net.Network)

    hiddenLayers = [2, 3, 4]  # [2, 10]
    Net2 = NeuralNetwork.UserDefineGraph(1, hiddenLayers, 2)
    # print(Net2.NeuralNet['Species'])
    Net2.UpdateGraph([1])
    # print(Net2.NeuralNet['Network'][0])
    # print(Net2.NeuralNet['Network'][0])
    # print(Net2.NeuralNet['Network'][-1]['Output'])

    print('----')
    print(Net2.Network[0])
    print(Net2.Network[1])
    print(Net2.Network[2])
    print(Net2.Network[3])
    print(Net2.Network[4])
    print(Net2.Network[5])
    print(Net2.Network[6])
    print(Net2.Network[7])
    print(Net2.Network[8])
    print(Net2.Network[9])
    print(Net2.Network[10])

    Net2.LogGraph('', 'Net2')
