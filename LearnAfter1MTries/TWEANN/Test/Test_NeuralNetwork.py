import unittest
from LearnAfter1MTries.TWEANN.NeuralNetwork import NeuralNetwork, Neuron
import math
import numpy as np


class NeuralNetworkTestCase(unittest.TestCase):
    def test_Neuron_ActivationFunction_sigmoid(self):
        actual = [Neuron.Sigmoid(u * 0.01) for u in range(0, 10)]
        expected = [1 / (1 + math.exp(-u * 0.01)) for u in range(0, 10)]
        self.assertEqual(expected, actual)

    def test_Neuron_ActivationFunction_relu(self):
        actual = [Neuron.Relu(u) for u in range(-10, 10)]
        expected = [max(0, u) for u in range(-10, 10)]
        self.assertEqual(expected, actual)

    def test_Neuron_ActivationFunction_tanh(self):
        actual = [Neuron.Tanh(u) for u in range(-10, 10)]
        expected = [math.tanh(u) for u in range(-10, 10)]
        self.assertEqual(expected, actual)

    def test_Neuron_ActivationFunction_leakyRelu(self):
        actual = [Neuron.LeakyRelu(u) for u in range(-10, 10)]
        expected = [max(-0.1, u) for u in range(-10, 10)]
        self.assertEqual(expected, actual)

    def test_Neuron_ActivationFunction_linear(self):
        actual = [Neuron.Linear(u) for u in range(-10, 10)]
        expected = [u for u in range(-10, 10)]
        self.assertEqual(expected, actual)

    def test_Neuron_UpdateNeuron_InputVectorGreaterThanWeightVector(self):
        testNeuron = Neuron()
        x = [1, 2, 3]
        w = [1, 2]
        b = 1
        rw = 1
        oldOut = 0

        testNeuron.neuron['Inputs'] = [x[0], x[1], x[2]]
        testNeuron.neuron['Weights'] = [w[0], w[1]]
        testNeuron.neuron['Bias'] = b
        testNeuron.neuron['Enabled'] = 1
        testNeuron.neuron['Recurrent'] = 1
        testNeuron.neuron['RecurrentWeight'] = rw
        testNeuron.neuron['Activation'] = 'linear'
        testNeuron.neuron['Output'] = oldOut

        testNeuron.UpdateNeuron()
        actual = testNeuron.neuron['Output']
        expected = x[0] * w[0] + x[1] * w[1] + x[2] * 0 + b + oldOut * rw

        self.assertEqual(expected, actual)

    def test_Neuron_UpdateNeuron_InputVectorLessThanWeightVector(self):
        testNeuron = Neuron()
        x = [1, 2, 3]
        w = [1, 2, 3]
        b = 1
        rw = 1
        oldOut = 0

        testNeuron.neuron['Inputs'] = [x[0], x[1]]
        testNeuron.neuron['Weights'] = [w[0], w[1], w[2]]
        testNeuron.neuron['Bias'] = b
        testNeuron.neuron['Enabled'] = 1
        testNeuron.neuron['Recurrent'] = 1
        testNeuron.neuron['RecurrentWeight'] = rw
        testNeuron.neuron['Activation'] = 'linear'
        testNeuron.neuron['Output'] = oldOut

        testNeuron.UpdateNeuron()
        actual = testNeuron.neuron['Output']
        expected = x[0] * w[0] + x[1] * w[1] + 0 * w[2] + b + oldOut * rw

        self.assertEqual(expected, actual)

    def test_NeuralNetwork_CreateInitialGraph_CheckOutput(self):
        numInputs = 6
        numOutputs = 4
        testNeuralNetwork = NeuralNetwork(numInputs, numOutputs)
        testNeuralNetwork.CreateInitialGraph()

        actual = testNeuralNetwork.NeuralNet['Output']
        expected = []

        self.assertEqual(expected, actual)

    def test_NeuralNetwork_CreateInitialGraph_CheckLayers(self):
        numInputs = 6
        numOutputs = 4
        testNeuralNetwork = NeuralNetwork(numInputs, numOutputs)
        testNeuralNetwork.CreateInitialGraph()

        actual = testNeuralNetwork.NeuralNet['Layers']
        expected = 0

        self.assertEqual(expected, actual)

    def test_NeuralNetwork_CreateInitialGraph_CheckSpecies(self):
        numInputs = 6
        numOutputs = 4
        testNeuralNetwork = NeuralNetwork(numInputs, numOutputs)
        testNeuralNetwork.CreateInitialGraph()

        actual = testNeuralNetwork.NeuralNet['Species']
        expected = [0, numOutputs]

        self.assertEqual(expected, actual)

    def test_NeuralNetwork_CreateInitialGraph_CheckMaxNodes(self):
        numInputs = 6
        numOutputs = 4
        testNeuralNetwork = NeuralNetwork(numInputs, numOutputs)
        testNeuralNetwork.CreateInitialGraph()

        actual = testNeuralNetwork.NeuralNet['MaxNodes']
        expected = 6

        self.assertEqual(expected, actual)

    def test_NeuralNetwork_CreateInitialGraph_CheckFitness(self):
        numInputs = 6
        numOutputs = 4
        testNeuralNetwork = NeuralNetwork(numInputs, numOutputs)
        testNeuralNetwork.CreateInitialGraph()

        actual = testNeuralNetwork.NeuralNet['Fitness']
        expected = None

        self.assertEqual(expected, actual)

    def test_NeuralNetwork_CreateInitialGraph_CheckNetwork_IDs(self):
        numInputs = 6
        numOutputs = 4
        testNeuralNetwork = NeuralNetwork(numInputs, numOutputs)
        testNeuralNetwork.CreateInitialGraph()

        key = 'ID'
        actualData = {key: []}
        for neuron in testNeuralNetwork.NeuralNet['Network']:
            actualData[key].append(neuron[key])

        actual = actualData[key]
        expected = list(range(0, numOutputs))

        self.assertEqual(expected, actual)

    def test_NeuralNetwork_CreateInitialGraph_CheckNetwork_Weight(self):
        numInputs = 6
        numOutputs = 4
        testNeuralNetwork = NeuralNetwork(numInputs, numOutputs)
        testNeuralNetwork.CreateInitialGraph()

        key = 'Weights'
        actualData = {key: []}
        for neuron in testNeuralNetwork.NeuralNet['Network']:
            actualData[key].append([neuron[key]])

        actual = [len(element[0]) for element in actualData[key]]
        expected = [numInputs] * numOutputs

        self.assertEqual(expected, actual)

    def test_NeuralNetwork_CreateInitialGraph_CheckNetwork_Bias(self):
        numInputs = 6
        numOutputs = 4
        testNeuralNetwork = NeuralNetwork(numInputs, numOutputs)
        testNeuralNetwork.CreateInitialGraph()

        key = 'Bias'
        actualData = {key: []}
        for neuron in testNeuralNetwork.NeuralNet['Network']:
            actualData[key].append(neuron[key])

        actual = [len([element]) for element in actualData[key]]
        expected = [1] * numOutputs

        self.assertEqual(expected, actual)

    def test_NeuralNetwork_CreateInitialGraph_CheckNetwork_Enabled(self):
        numInputs = 6
        numOutputs = 4
        testNeuralNetwork = NeuralNetwork(numInputs, numOutputs)
        testNeuralNetwork.CreateInitialGraph()

        key = 'Enabled'
        actualData = {key: []}
        for neuron in testNeuralNetwork.NeuralNet['Network']:
            actualData[key].append(neuron[key])

        actual = [(element in [0, 1]) for element in actualData[key]]
        expected = [True] * numOutputs

        self.assertEqual(expected, actual)

    def test_NeuralNetwork_CreateInitialGraph_CheckNetwork_Recurrent(self):
        numInputs = 6
        numOutputs = 4
        testNeuralNetwork = NeuralNetwork(numInputs, numOutputs)
        testNeuralNetwork.CreateInitialGraph()

        key = 'Recurrent'
        actualData = {key: []}
        for neuron in testNeuralNetwork.NeuralNet['Network']:
            actualData[key].append(neuron[key])

        actual = [(element in [0, 1]) for element in actualData[key]]
        expected = [True] * numOutputs

        self.assertEqual(expected, actual)

    def test_NeuralNetwork_CreateInitialGraph_CheckNetwork_RecurrentWeight(self):
        numInputs = 6
        numOutputs = 4
        testNeuralNetwork = NeuralNetwork(numInputs, numOutputs)
        testNeuralNetwork.CreateInitialGraph()

        key = 'RecurrentWeight'
        actualData = {key: []}
        for neuron in testNeuralNetwork.NeuralNet['Network']:
            actualData[key].append(neuron[key])

        actual = [len([element]) for element in actualData[key]]
        expected = [1] * numOutputs

        self.assertEqual(expected, actual)

    def test_NeuralNetwork_CreateInitialGraph_CheckNetwork_Activation(self):
        numInputs = 6
        numOutputs = 4
        testNeuralNetwork = NeuralNetwork(numInputs, numOutputs)
        testNeuralNetwork.CreateInitialGraph()

        activationList = Neuron().activationFunc

        key = 'Activation'
        actualData = {key: []}
        for neuron in testNeuralNetwork.NeuralNet['Network']:
            actualData[key].append(neuron[key])

        actual = [(element in activationList) for element in actualData[key]]
        expected = [True] * numOutputs

        self.assertEqual(expected, actual)

    def test_NeuralNetwork_CreateInitialGraph_CheckNetwork_NodeNum(self):
        numInputs = 6
        numOutputs = 4
        testNeuralNetwork = NeuralNetwork(numInputs, numOutputs)
        testNeuralNetwork.CreateInitialGraph()

        key = 'NodeNum'
        actualData = {key: []}
        for neuron in testNeuralNetwork.NeuralNet['Network']:
            actualData[key].append(neuron[key])

        actual = actualData[key]
        expected = list(range(0, numOutputs))

        self.assertEqual(expected, actual)

    def test_NeuralNetwork_CreateInitialGraph_CheckNetwork_InputVector(self):
        numInputs = 6
        numOutputs = 4
        testNeuralNetwork = NeuralNetwork(numInputs, numOutputs)
        testNeuralNetwork.CreateInitialGraph()

        key = 'Inputs'
        actualData = {key: []}
        for neuron in testNeuralNetwork.NeuralNet['Network']:
            actualData[key].append(neuron[key])

        actual = [element for element in actualData[key]]
        expected = [[]] * numOutputs

        self.assertEqual(expected, actual)

    def test_NeuralNetwork_CreateInitialGraph_CheckNetwork_OutputVector(self):
        numInputs = 6
        numOutputs = 4
        testNeuralNetwork = NeuralNetwork(numInputs, numOutputs)
        testNeuralNetwork.CreateInitialGraph()

        key = 'Output'
        actualData = {key: []}
        for neuron in testNeuralNetwork.NeuralNet['Network']:
            actualData[key].append(neuron[key])

        actual = [element for element in actualData[key]]
        expected = [0] * numOutputs

        self.assertEqual(expected, actual)

    def test_NeuralNetwork__UserDefineGraph_CheckOutput(self):
        numInputs = 6
        numOutputs = 4
        hiddenLayers = [5, 3, 4]
        testNeuralNetwork = NeuralNetwork.UserDefineGraph(numInputs, hiddenLayers, numOutputs)

        actual = testNeuralNetwork.NeuralNet['Output']
        expected = []

        self.assertEqual(expected, actual)

    def test_NeuralNetwork__UserDefineGraph_CheckLayers(self):
        numInputs = 6
        numOutputs = 4
        hiddenLayers = [5, 3, 4]
        testNeuralNetwork = NeuralNetwork.UserDefineGraph(numInputs, hiddenLayers, numOutputs)

        actual = testNeuralNetwork.NeuralNet['Layers']
        expected = len(hiddenLayers) + 1

        self.assertEqual(expected, actual)

    def test_NeuralNetwork__UserDefineGraph_CheckSpecies(self):
        numInputs = 6
        numOutputs = 4
        hiddenLayers = [5, 3, 4]
        testNeuralNetwork = NeuralNetwork.UserDefineGraph(numInputs, hiddenLayers, numOutputs)

        actual = testNeuralNetwork.NeuralNet['Species']
        expected = hiddenLayers + [numOutputs]

        self.assertEqual(expected, actual)

    def test_NeuralNetwork__UserDefineGraph_CheckMaxNodes(self):
        numInputs = 6
        numOutputs = 4
        hiddenLayers = [5, 3, 4]
        testNeuralNetwork = NeuralNetwork.UserDefineGraph(numInputs, hiddenLayers, numOutputs)

        actual = testNeuralNetwork.NeuralNet['MaxNodes']
        expected = len(hiddenLayers) - 1 + numOutputs

        self.assertEqual(expected, actual)

    def test_NeuralNetwork__UserDefineGraph_CheckFitness(self):
        numInputs = 6
        numOutputs = 4
        hiddenLayers = [5, 3, 4]
        testNeuralNetwork = NeuralNetwork.UserDefineGraph(numInputs, hiddenLayers, numOutputs)

        actual = testNeuralNetwork.NeuralNet['Fitness']
        expected = None

        self.assertEqual(expected, actual)

    def test_NeuralNetwork__UserDefineGraph_CheckNetwork_IDs(self):
        numInputs = 6
        numOutputs = 4
        hiddenLayers = [5, 3, 4]
        testNeuralNetwork = NeuralNetwork.UserDefineGraph(numInputs, hiddenLayers, numOutputs)

        key = 'ID'
        actualData = {key: []}
        for neuron in testNeuralNetwork.NeuralNet['Network']:
            actualData[key].append(neuron[key])

        actual = actualData[key]
        maxNodes = sum(hiddenLayers) + numOutputs
        expected = list(range(0, maxNodes))

        self.assertEqual(expected, actual)

    def test_NeuralNetwork__UserDefineGraph_CheckNetwork_Weight(self):
        numInputs = 6
        numOutputs = 4
        hiddenLayers = [5, 3, 4]
        testNeuralNetwork = NeuralNetwork.UserDefineGraph(numInputs, hiddenLayers, numOutputs)

        key = 'Weights'
        actualData = {key: []}
        for neuron in testNeuralNetwork.NeuralNet['Network']:
            actualData[key].append([neuron[key]])

        actual = [len(element[0]) for element in actualData[key]]
        hiddenWeightsNum = []
        for hid in hiddenLayers:
            hiddenWeightsNum += [hid] * hid
        expected = hiddenWeightsNum + [numOutputs] * numOutputs

        self.assertEqual(expected, actual)

    def test_NeuralNetwork__UserDefineGraph_CheckNetwork_Bias(self):
        numInputs = 6
        numOutputs = 4
        hiddenLayers = [5, 3, 4]
        testNeuralNetwork = NeuralNetwork.UserDefineGraph(numInputs, hiddenLayers, numOutputs)

        key = 'Bias'
        actualData = {key: []}
        for neuron in testNeuralNetwork.NeuralNet['Network']:
            actualData[key].append(neuron[key])

        actual = [len([element]) for element in actualData[key]]
        maxNodes = sum(hiddenLayers) + numOutputs
        expected = [1] * maxNodes

        self.assertEqual(expected, actual)

    def test_NeuralNetwork__UserDefineGraph_CheckNetwork_Enabled(self):
        numInputs = 6
        numOutputs = 4
        hiddenLayers = [5, 3, 4]
        testNeuralNetwork = NeuralNetwork.UserDefineGraph(numInputs, hiddenLayers, numOutputs)

        key = 'Enabled'
        actualData = {key: []}
        for neuron in testNeuralNetwork.NeuralNet['Network']:
            actualData[key].append(neuron[key])

        actual = [(element in [0, 1]) for element in actualData[key]]
        maxNodes = sum(hiddenLayers) + numOutputs
        expected = [True] * maxNodes

        self.assertEqual(expected, actual)

    def test_NeuralNetwork__UserDefineGraph_CheckNetwork_Recurrent(self):
        numInputs = 6
        numOutputs = 4
        hiddenLayers = [5, 3, 4]
        testNeuralNetwork = NeuralNetwork.UserDefineGraph(numInputs, hiddenLayers, numOutputs)

        key = 'Recurrent'
        actualData = {key: []}
        for neuron in testNeuralNetwork.NeuralNet['Network']:
            actualData[key].append(neuron[key])

        actual = [(element in [0, 1]) for element in actualData[key]]
        maxNodes = sum(hiddenLayers) + numOutputs
        expected = [True] * maxNodes

        self.assertEqual(expected, actual)

    def test_NeuralNetwork__UserDefineGraph_CheckNetwork_RecurrentWeight(self):
        numInputs = 6
        numOutputs = 4
        hiddenLayers = [5, 3, 4]
        testNeuralNetwork = NeuralNetwork.UserDefineGraph(numInputs, hiddenLayers, numOutputs)

        key = 'RecurrentWeight'
        actualData = {key: []}
        for neuron in testNeuralNetwork.NeuralNet['Network']:
            actualData[key].append(neuron[key])

        actual = [len([element]) for element in actualData[key]]
        maxNodes = sum(hiddenLayers) + numOutputs
        expected = [1] * maxNodes

        self.assertEqual(expected, actual)

    def test_NeuralNetwork__UserDefineGraph_CheckNetwork_Activation(self):
        numInputs = 6
        numOutputs = 4
        hiddenLayers = [5, 3, 4]
        testNeuralNetwork = NeuralNetwork.UserDefineGraph(numInputs, hiddenLayers, numOutputs)

        activationList = Neuron().activationFunc

        key = 'Activation'
        actualData = {key: []}
        for neuron in testNeuralNetwork.NeuralNet['Network']:
            actualData[key].append(neuron[key])

        actual = [(element in activationList) for element in actualData[key]]
        maxNodes = sum(hiddenLayers) + numOutputs
        expected = [True] * maxNodes

        self.assertEqual(expected, actual)

    def test_NeuralNetwork__UserDefineGraph_CheckNetwork_NodeNum(self):
        numInputs = 6
        numOutputs = 4
        hiddenLayers = [5, 3, 4]
        testNeuralNetwork = NeuralNetwork.UserDefineGraph(numInputs, hiddenLayers, numOutputs)

        key = 'NodeNum'
        actualData = {key: []}
        for neuron in testNeuralNetwork.NeuralNet['Network']:
            actualData[key].append(neuron[key])

        actual = actualData[key]
        hiddenNodes = []
        for hid in hiddenLayers:
            hiddenNodes += list(range(0, hid))

        expected = hiddenNodes + list(range(0, numOutputs))
        self.assertEqual(expected, actual)

    def test_NeuralNetwork__UserDefineGraph_CheckNetwork_InputVector(self):
        numInputs = 6
        numOutputs = 4
        hiddenLayers = [5, 3, 4]
        testNeuralNetwork = NeuralNetwork.UserDefineGraph(numInputs, hiddenLayers, numOutputs)

        key = 'Inputs'
        actualData = {key: []}
        for neuron in testNeuralNetwork.NeuralNet['Network']:
            actualData[key].append(neuron[key])

        actual = [element for element in actualData[key]]
        maxNodes = sum(hiddenLayers) + numOutputs
        expected = [[]] * maxNodes

        self.assertEqual(expected, actual)

    def test_NeuralNetwork__UserDefineGraph_CheckNetwork_OutputVector(self):
        numInputs = 6
        numOutputs = 4
        hiddenLayers = [5, 3, 4]
        testNeuralNetwork = NeuralNetwork.UserDefineGraph(numInputs, hiddenLayers, numOutputs)

        key = 'Output'
        actualData = {key: []}
        for neuron in testNeuralNetwork.NeuralNet['Network']:
            actualData[key].append(neuron[key])

        actual = [element for element in actualData[key]]
        maxNodes = sum(hiddenLayers) + numOutputs
        expected = [0] * maxNodes

        self.assertEqual(expected, actual)

    def test_NeuralNetwork_UpdateGraph(self):
        numInputs = 2
        numOutputs = 2
        hiddenLayers = [2, 1, 3]
        X = [1, 2]
        w00 = [1, 2]
        w01 = [2, 4]

        w10 = [5]

        w20 = [5, 6, 7]
        w21 = [8, 9, 10]
        w22 = [11, 12, 13]

        o0 = [14, 15]
        o1 = [16, 17]

        weight = [w00] + [w01] + [w10] + [w20] + [w21] + [w22] + [o0] + [o1]

        b = 1
        r = 1
        testNeuralNetwork = NeuralNetwork.UserDefineGraph(numInputs, hiddenLayers, numOutputs)
        maxNodes = sum(hiddenLayers) + numOutputs

        for neuron in range(0, maxNodes):
            testNeuralNetwork.NeuralNet['Network'][neuron]['Bias'] = 1
            testNeuralNetwork.NeuralNet['Network'][neuron]['Recurrent'] = 1
            testNeuralNetwork.NeuralNet['Network'][neuron]['RecurrentWeight'] = 1
            testNeuralNetwork.NeuralNet['Network'][neuron]['Weights'] = weight[neuron]

        testNeuralNetwork.UpdateGraph(X)

        # Layer 0 output
        y00 = X[0] * w00[0] + X[1] * w00[1] + b
        y01 = X[0] * w01[0] + X[1] * w01[1] + b

        # Layer 1 output
        y10 = y00 * w10[0] + b

        # Layer 2 output
        y20 = y10 * w20[0] + b
        y21 = y10 * w21[0] + b
        y22 = y10 * w22[0] + b

        #  output
        yo0 = y20 * o0[0] + y21 * o0[1] + b
        yo1 = y20 * o1[0] + y21 * o1[1] + b

        expected = [y00, y01, y10, y20, y21, y22, yo0, yo1]

        testNeuralNetwork.UpdateGraph(X)

        # Layer 0 output
        y00 = X[0] * w00[0] + X[1] * w00[1] + b + y00 * r
        y01 = X[0] * w01[0] + X[1] * w01[1] + b + y01 * r

        # Layer 1 output
        y10 = y00 * w10[0] + b + y10 * r

        # Layer 2 output
        y20 = y10 * w20[0] + b + y20 * r
        y21 = y10 * w21[0] + b + y21 * r
        y22 = y10 * w22[0] + b + y22 * r

        #  output
        yo0 = y20 * o0[0] + y21 * o0[1] + b + yo0 * r
        yo1 = y20 * o1[0] + y21 * o1[1] + b + yo1 * r

        expected = [y00, y01, y10, y20, y21, y22, yo0, yo1]

        actualData = {'Output': []}
        for neuron in testNeuralNetwork.NeuralNet['Network']:
            actualData['Output'].append(neuron['Output'])

        actual = actualData['Output']
        self.assertEqual(expected, actual)

    def test_NeuralNetwork_FindNeuronsInLayer(self):
        numInputs = 6
        numOutputs = 4
        hiddenLayers = [5, 3, 4]
        testNeuralNetwork = NeuralNetwork.UserDefineGraph(numInputs, hiddenLayers, numOutputs)

        neuronsInLayer0 = testNeuralNetwork.FindNeuronsInLayer(0)
        neuronsInLayer1 = testNeuralNetwork.FindNeuronsInLayer(1)
        neuronsInLayer2 = testNeuralNetwork.FindNeuronsInLayer(2)
        neuronsInOutput = testNeuralNetwork.FindNeuronsInLayer(-1)

        expected = neuronsInLayer0 + neuronsInLayer1 + neuronsInLayer2 + neuronsInOutput
        actual = testNeuralNetwork.NeuralNet['Network']

        self.assertEqual(expected, actual)

    def test_NeuralNetwork_AddNeuron(self):
        numInputs = 6
        numOutputs = 4
        hiddenLayers = [5, 3, 4]
        testNeuralNetwork = NeuralNetwork.UserDefineGraph(numInputs, hiddenLayers, numOutputs)

        newNeuron = Neuron()
        newNeuron.neuron['Layer'] = 2
        newNeuron.neuron['ID'] = 2

        testNeuralNetwork.AddNeuron(newNeuron.neuron)
        actual = testNeuralNetwork.NeuralNet['Network'][16]
        expected = newNeuron.neuron
        expected['ID'] = sum(hiddenLayers) + numOutputs
        expected['NodeNum'] = hiddenLayers[newNeuron.neuron['Layer'] - 1] + 1

        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
