import math
import json
import numpy as np
from random import random
from random import choice
from LearnAfter1MTries.TWEANN import TweannRender
import copy

class Neuron:
    def __init__(self):
        self.neuron = {
            'ID': [],
            'Weights': [],
            'Bias': [],
            'Enabled': 1,
            'Recurrent': [],
            'RecurrentWeight': [],
            'Layer': [],
            'Activation': [],
            'NodeNum': 0,
            'Inputs': [],
            'Output': 0,
        }
        self.activationFunc = ['sigmoid', 'relu', 'tanh', 'leakyRelu', 'linear']

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
        return max(-0.1, u)

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

        #Padding is being done so that inputs and weights vector lengths dont have to be compatible
        if diff > 0:
            weights = np.pad(weights, (0, diff), 'constant', constant_values=(0, 0))
        if diff < 0:
            inputs = np.pad(inputs, (0, -diff), 'constant', constant_values=(0, 0))

        bias = np.array(self.neuron['Bias'])
        Out = np.array(self.neuron['Output'])
        recurrent = np.array(self.neuron['Recurrent'])
        recurrentWeight = np.array(self.neuron['RecurrentWeight'])

        Output = inputs.dot(weights.transpose()) + bias + Out * recurrent * recurrentWeight
        # print('Output {}'.format(Output))

        if self.neuron['Activation'] == 'sigmoid':
            self.neuron['Output'] = self.Sigmoid(Output)

        if self.neuron['Activation'] == 'relu':
            self.neuron['Output'] = self.Relu(Output)

        if self.neuron['Activation'] == 'tanh':
            self.neuron['Output'] = self.Tanh(Output)

        if self.neuron['Activation'] == 'leakyRelu':
            self.neuron['Output'] = self.LeakyRelu(Output)

        if self.neuron['Activation'] == 'linear':
            self.neuron['Output'] = self.Linear(Output)

        # print('Act Output {}'.format(self.neuron['Output']))


class NeuralNetwork:
    def __init__(self, numInputs, numOutputs):
        self.numInputs = numInputs
        self.numOutputs = numOutputs

        self.Neuron = Neuron()
        self.NeuralNet = {
            'Network': [],
            'Output': 0,
            'Layers': 0,
            'Species': [-1, -1],  # [ num hiddenlayers, layers]
            'MaxNodes': None,
            'Fitness': None
        }

    def __str__(self):
        return '<%s>' % self.NeuralNet

    def CreateInitialGraph(self):
        # Each Input connected to each output neuron and it activation function is a sigmoid
        for outs in range(0, self.numOutputs):
            neuron = self.Neuron.neuron
            neuron['Layer'] = 0
            neuron['Activation'] = choice(self.Neuron.activationFunc)
            neuron['ID'] = outs
            neuron['NodeNum'] = outs
            neuron['Weights'] = [random() for inputs in range(0, self.numInputs)]
            neuron['Bias'] = random()
            neuron['Recurrent'] = choice([0, 1])
            neuron['RecurrentWeight'] = random()
            neuron['Enabled'] = choice([0, 1])
            neuron['Output'] = 0
            self.NeuralNet['Network'].append(neuron)
            neuron = []

        self.NeuralNet['Species'] = [0, self.numOutputs]
        self.NeuralNet['MaxNodes'] = max(self.numOutputs, self.numInputs)
        self.NeuralNet['Fitness'] = 0

    def UpdateGraph(self, NetInputs):
        nextLayerInput = []
        nextnextLayerInput = []
        layerList = [neuron['Layer'] for neuron in self.NeuralNet['Network']]

        Iter = 0
        for index, neuron in enumerate(self.NeuralNet['Network']):
            if neuron['Layer'] == 0:
                neuron['Inputs'] = NetInputs
                self.Neuron.UpdateNeuron()
                neuron['Output'] = self.Neuron.neuron['Output']
                self.NeuralNet['Output'] = neuron['Output']
                nextLayerInput.append(self.Neuron.neuron['Output'])
            else:
                if neuron['Layer'] == layerList[index]:
                    neuron['Inputs'] = nextLayerInput.copy()
                    self.Neuron.UpdateNeuron()
                    neuron['Output'] = self.Neuron.neuron['Output']
                    self.NeuralNet['Output'] = neuron['Output']
                    nextnextLayerInput.append(self.Neuron.neuron['Output'])
                    Iter += 1
                    if layerList.count(layerList[index]) == Iter:
                        Iter = 0
                        nextLayerInput = nextnextLayerInput.copy()
                        nextnextLayerInput = []

    def FindNeuronsInLayer(self,layer):
        neuronInLayer = []
        for neuron in self.NeuralNet['Network']:
            if neuron['Layer'] == layer:
                neuronInLayer.append(neuron)
        return neuronInLayer

    def AddNeuron(self, newNeuron):

        addedNeuron = copy.deepcopy(newNeuron)
        newNodeNum = -1
        newNodeID = -1
        for neuron in self.NeuralNet['Network']:
            if neuron['Layer'] == addedNeuron['Layer']:
                if neuron['NodeNum'] > newNodeNum:
                    newNodeNum = neuron['NodeNum']

                if neuron['ID'] > newNodeID:
                    newNodeID = neuron['ID']

        addedNeuron['NodeNum'] = newNodeNum + 1
        addedNeuron['ID'] = newNodeID + 1
        addedNeuron['Species'] = [addedNeuron['Layer'], addedNeuron['ID']]
        self.NeuralNet['Network'].append(addedNeuron)
        
    @staticmethod
    def UserDefineGraph(numInputs, HiddenLayers, numOutputs, activation='relu', recurrent=0):
        network = NeuralNetwork(numInputs, numOutputs)
        neuron = network.Neuron.neuron
        species = HiddenLayers.copy()
        species.append(numOutputs)
        ID = 0
        for layer, nodes in enumerate(HiddenLayers):
            neuron['Layer'] = layer
            for nodeNum, node in enumerate(range(0, nodes)):
                neuron['ID'] = ID
                neuron['Weights'] = [random() for inputs in range(0, nodes)]
                neuron['Bias'] = random()
                neuron['Activation'] = activation
                neuron['Recurrent'] = recurrent
                neuron['RecurrentWeight'] = random()
                neuron['Enabled'] = 1
                neuron['Output'] = 0
                neuron['NodeNum'] = nodeNum
                ID += 1
                network.NeuralNet['Network'].append(neuron.copy())

        for outs in range(0, numOutputs):
            neuron['Layer'] = -1
            neuron['ID'] = ID
            neuron['Weights'] = [random() for inputs in range(0, nodes)]
            neuron['Bias'] = random()
            neuron['Activation'] = activation
            neuron['Recurrent'] = recurrent
            neuron['RecurrentWeight'] = random()
            neuron['Enabled'] = 1
            neuron['Output'] = 0
            neuron['NodeNum'] = outs
            ID += 1
            network.NeuralNet['Network'].append(neuron.copy())

        network.NeuralNet['Layers'] = len(HiddenLayers) + 1
        network.NeuralNet['Species'] = species.copy()
        species.insert(0, numInputs)
        network.NeuralNet['MaxNodes'] = max(species)
        return network

    def DrawGraph(self, graphWindow, tileSize=64, offset=0):
        inputOffset = (self.NeuralNet['MaxNodes'] - self.numInputs) / 2
        for inputs in range(0, self.numInputs):
            TweannRender.Inputs(graphWindow, 0, inputs, tileSize, offset, inputOffset)

        for node in self.NeuralNet['Network']:

            if node['Layer'] != -1:
                nodeXpos = node['Layer'] + 1
            else:
                nodeXpos = self.NeuralNet['Layers']

            elements = self.NeuralNet['Species'][node['Layer']]
            nodeOffset = (self.NeuralNet['MaxNodes'] - elements) / 2

            TweannRender.Perceptron(graphWindow, nodeXpos,
                                    node['NodeNum'], node['Enabled'], node['Activation'], node['Recurrent'],
                                    tileSize, offset, nodeOffset)

    def InternalDrawGraph(self):
        graphWindow = TweannRender.GraphWindow()
        self.DrawGraph(graphWindow, tileSize=64, offset=4)

        graphWindow.draw()
        graphWindow.eventManager()

    def LogGraph(self, directory, filename):
        with open(directory + filename + '.json', 'w') as json_file:
            json.dump(self.NeuralNet, json_file, indent=4)


if __name__ == '__main__':
    Net = NeuralNetwork(2, 3)
    ##print(Net.Network[1])
    #print('-----')
    #Net.UpdateGraph([1, 1])
    #print(Net.Network)

    #Net.CreateInitialGraph()
    # Net.DrawGraph()

    hiddenLayers = [5, 3, 4]  # [2, 10]
    Net2 = NeuralNetwork.UserDefineGraph(3, hiddenLayers, 5)
    # print(Net2.NeuralNet['Species'])
    Net2.UpdateGraph([1, 1, 1])
    # print(Net2.NeuralNet['Network'][0])
    # print(Net2.NeuralNet['Network'][0])
    # print(Net2.NeuralNet['Network'][-1]['Output'])

    # print('----')
    # print(Net2.Network[0])
    # print(Net2.Network[1])
    # print(Net2.Network[2])
    # print(Net2.Network[3])
    # print(Net2.Network[4])
    # print(Net2.Network[5])
    # print(Net2.Network[6])
    # print(Net2.Network[7])
    # print(Net2.Network[8])
    # print(Net2.Network[9])
    # print(Net2.Network[10])

    # Net2.LogGraph('', 'Net2')
    while True:
        Net2.InternalDrawGraph()
