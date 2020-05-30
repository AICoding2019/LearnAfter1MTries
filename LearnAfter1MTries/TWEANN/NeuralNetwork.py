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
        try:
            y = 1 / (1 + math.exp(-u))
        except OverflowError:
            if u > 0:
                y = 1
            else:
                y=0
        return y

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

        # Padding is being done so that inputs and weights vector lengths dont have to be compatible
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

        self.neuron['Output'] = self.neuron['Output'] * self.neuron['Enabled']


class NeuralNetwork:
    def __init__(self, numInputs, numOutputs):
        self.numInputs = numInputs
        self.numOutputs = numOutputs

        self.NeuralNet = {
            'Network': [],
            'Output': [],
            'Layers': 0,
            'Species': [self.numInputs, 0, self.numOutputs],  # [inputs, hiddenlayers, outputs]
            'MaxNodes': None,
            'Fitness': None
        }

    def __str__(self):
        return '<%s>' % self.NeuralNet

    def CreateInitialGraph(self):
        # Each Input connected to each output neuron and it activation function is a sigmoid
        for outs in range(0, self.numOutputs):
            newNeuron = Neuron()
            neuron = newNeuron.neuron
            neuron['Layer'] = -1
            neuron['Activation'] = choice(newNeuron.activationFunc)
            neuron['ID'] = outs
            neuron['NodeNum'] = outs
            neuron['Weights'] = [random() for _ in range(0, self.numInputs)]
            neuron['Bias'] = random()
            neuron['Recurrent'] = choice([0, 1])
            neuron['RecurrentWeight'] = random()
            neuron['Enabled'] = choice([0, 1])
            neuron['Output'] = 0
            self.NeuralNet['Network'].append(copy.deepcopy(neuron))

        self.NeuralNet['Layers'] = 0
        self.NeuralNet['Species'] = [self.numInputs, 0, self.numOutputs]
        self.NeuralNet['MaxNodes'] = max(self.NeuralNet['Species'])
        self.NeuralNet['Fitness'] = None

    def UpdateGraph(self, NetInputs):
        nextLayerInput = []
        nextnextLayerInput = []

        layerList = [neuron['Layer'] for neuron in self.NeuralNet['Network']]

        Iter = 0
        for index, neuron in enumerate(self.NeuralNet['Network']):
            NeuralNetNeuron = Neuron()
            NeuralNetNeuron.neuron = neuron
            if neuron['Layer'] == 0:
                neuron['Inputs'] = NetInputs
                NeuralNetNeuron.UpdateNeuron()
                nextLayerInput.append(NeuralNetNeuron.neuron['Output'])
            else:
                if neuron['Layer'] == layerList[index]:
                    neuron['Inputs'] = copy.deepcopy(nextLayerInput)
                    NeuralNetNeuron.UpdateNeuron()
                    nextnextLayerInput.append(NeuralNetNeuron.neuron['Output'])
                    Iter += 1
                    if layerList.count(layerList[index]) == Iter:
                        Iter = 0
                        nextLayerInput = copy.deepcopy(nextnextLayerInput)
                        nextnextLayerInput = []
                if neuron['Layer'] == -1:
                    self.NeuralNet['Output'].append(copy.deepcopy(NeuralNetNeuron.neuron['Output']))

    def FindNeuronsInLayer(self, layer):
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

        if len(self.NeuralNet['Species']) - 2 > addedNeuron['Layer']:
            self.NeuralNet['Species'][addedNeuron['Layer'] + 1] = self.NeuralNet['Species'][
                                                                      addedNeuron['Layer'] + 1] + 1
        else:
            insertSpecies = len(self.NeuralNet['Species']) - 1
            self.NeuralNet['Species'].insert(insertSpecies, 1)
            addedNeuron['Layer'] = len(self.NeuralNet['Species']) - 3

        if addedNeuron['Layer'] > self.NeuralNet['Layers']:
            self.NeuralNet['Layers'] = addedNeuron['Layer']

        self.NeuralNet['MaxNodes'] = max(self.NeuralNet['Species'])

        self.NeuralNet['Network'].append(copy.deepcopy(addedNeuron))

    def findMaxNodes(self):
        maxNodes = -1
        for neuron in self.NeuralNet['Network']:
            if neuron['NodeNum'] > maxNodes:
                maxNodes = neuron['NodeNum']

        self.NeuralNet['MaxNodes'] = max(maxNodes, self.numInputs)

    @staticmethod
    def UserDefineGraph(numInputs, hiddenLayers, numOutputs, activation='relu', recurrent=0):
        network = NeuralNetwork(numInputs, numOutputs)
        newNeuron = Neuron()
        neuron = newNeuron.neuron
        species = [numInputs] + hiddenLayers + [numOutputs]

        if activation == 'random':
            activation = choice(newNeuron.activationFunc)
        ID = 0
        for layer, nodes in enumerate(hiddenLayers):
            neuron['Layer'] = layer
            for nodeNum, node in enumerate(range(0, nodes)):
                neuron['ID'] = ID
                neuron['Weights'] = [random() for _ in range(0, nodes)]
                neuron['Bias'] = random()
                neuron['Activation'] = activation
                neuron['Recurrent'] = recurrent
                neuron['RecurrentWeight'] = random()
                neuron['Enabled'] = 1
                neuron['Output'] = 0
                neuron['NodeNum'] = nodeNum
                ID += 1
                network.NeuralNet['Network'].append(copy.deepcopy(neuron))

        for outs in range(0, numOutputs):
            neuron['Layer'] = -1
            neuron['ID'] = ID
            neuron['Weights'] = [random() for _ in range(0, numOutputs)]
            neuron['Bias'] = random()
            neuron['Activation'] = activation
            neuron['Recurrent'] = recurrent
            neuron['RecurrentWeight'] = random()
            neuron['Enabled'] = 1
            neuron['Output'] = 0
            neuron['NodeNum'] = outs
            ID += 1
            network.NeuralNet['Network'].append(copy.deepcopy(neuron))

        network.NeuralNet['Layers'] = len(hiddenLayers) + 1
        network.NeuralNet['Species'] = species
        network.NeuralNet['MaxNodes'] = max(species)
        return network

    def DrawGraph(self, graphWindow, tileSize=24, offset=0):
        print('Drawing Species',self.NeuralNet['Species'])
        inputOffset = (self.NeuralNet['MaxNodes'] - self.numInputs) / 2
        for inputs in range(0, self.numInputs):
            TweannRender.Inputs(graphWindow, 0, inputs, tileSize, offset, inputOffset)

        for node in self.NeuralNet['Network']:
            if node['Layer'] != -1:
                nodeXpos = node['Layer'] + 1
                elements = self.NeuralNet['Species'][node['Layer'] + 1]
            else:
                nodeXpos = len(self.NeuralNet['Species']) - 1
                elements = self.NeuralNet['Species'][-1]

            nodeOffset = (self.NeuralNet['MaxNodes'] - elements) / 2

            TweannRender.Perceptron(graphWindow, nodeXpos,
                                    node['NodeNum'], node['Enabled'], node['Activation'], node['Recurrent'],
                                    tileSize, offset, nodeOffset)

    def InternalDrawGraph(self):
        graphWindow = TweannRender.GraphWindow()
        self.DrawGraph(graphWindow, tileSize=24, offset=0)
        print(f"Drawing best ever Species: {self.NeuralNet['Species']}")
        if self.NeuralNet['Species'][1] > 0:
            y = 0

        graphWindow.draw()
        graphWindow.eventManager()

    def LogGraph(self, directory, filename):
        with open(directory + filename + '.json', 'w') as json_file:
            json.dump(self.NeuralNet, json_file, indent=4)


if __name__ == '__main__':

    numInputs = 6
    numOutputs = 3
    hiddenLayers = [5, 3, 2, 1, 4]

    #    testNeuralNetwork = NeuralNetwork.UserDefineGraph(numInputs, hiddenLayers, numOutputs)
    testNeuralNetwork = NeuralNetwork(numInputs, numOutputs)
    testNeuralNetwork.CreateInitialGraph()
    # newNeuron = Neuron()
    # newNeuron.neuron['Layer'] = 0
    # newNeuron.neuron['ID'] = 2
    # newNeuron.neuron['Activation'] = 'tanh'
    # testNeuralNetwork.AddNeuron(newNeuron.neuron)

    print(testNeuralNetwork.NeuralNet['Species'])

    while True:
        testNeuralNetwork.InternalDrawGraph()
        y = 0
