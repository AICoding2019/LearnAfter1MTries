import unittest
from LearnAfter1MTries.TWEANN.NeuralNetwork import NeuralNetwork, Neuron
import math


class NeuralNetworkTestCase(unittest.TestCase):
    def test_Neuron_ActivationFunction_sigmoid(self):
        actual = [Neuron.Sigmoid(u * 0.01) for u in range(0, 10)]
        expected = [1 / (1 + math.exp(-u * 0.01)) for u in range(0, 10)]
        self.assertEqual(actual, expected)

    def test_Neuron_ActivationFunction_relu(self):
        actual = [Neuron.Relu(u) for u in range(-10, 10)]
        expected = [max(0, u) for u in range(-10, 10)]
        self.assertEqual(actual, expected)

    def test_Neuron_ActivationFunction_tanh(self):
        actual = [Neuron.Tanh(u) for u in range(-10, 10)]
        expected = [math.tanh(u) for u in range(-10, 10)]
        self.assertEqual(actual, expected)

    def test_Neuron_ActivationFunction_leakyRelu(self):
        actual = [Neuron.LeakyRelu(u) for u in range(-10, 10)]
        expected = [max(-0.1, u) for u in range(-10, 10)]
        self.assertEqual(actual, expected)

    def test_Neuron_ActivationFunction_linear(self):
        actual = [Neuron.Linear(u) for u in range(-10, 10)]
        expected = [u for u in range(-10, 10)]
        self.assertEqual(actual, expected)

    def test_Neuron_UpdateNeuron_InputVectorGreaterThanWeightVector(self):
        testNeuron = Neuron()
        x1 = 1
        x2 = 2
        x3 = 3
        w1 = 1
        w2 = 2
        b = 1
        rw = 1
        oldOut = 0

        testNeuron.neuron['Inputs'] = [x1, x2, x3]
        testNeuron.neuron['Weights'] = [w1, w2]
        testNeuron.neuron['Bias'] = b
        testNeuron.neuron['Enabled'] = 1
        testNeuron.neuron['Recurrent'] = 1
        testNeuron.neuron['RecurrentWeight'] = rw
        testNeuron.neuron['Activation'] = 'linear'
        testNeuron.neuron['Output'] = oldOut

        testNeuron.UpdateNeuron()

        expected = x1 * w1 + x2 * w2 + x3 * 0 + b + oldOut * rw
        self.assertEqual(expected, testNeuron.neuron['Output'])

    def test_Neuron_UpdateNeuron_InputVectorLessThanWeightVector(self):
        testNeuron = Neuron()
        x1 = 1
        x2 = 2
        w1 = 1
        w2 = 2
        w3 = 3
        b = 1
        rw = 1
        oldOut = 0

        testNeuron.neuron['Inputs'] = [x1, x2]
        testNeuron.neuron['Weights'] = [w1, w2, w3]
        testNeuron.neuron['Bias'] = b
        testNeuron.neuron['Enabled'] = 1
        testNeuron.neuron['Recurrent'] = 1
        testNeuron.neuron['RecurrentWeight'] = rw
        testNeuron.neuron['Activation'] = 'linear'
        testNeuron.neuron['Output'] = oldOut

        testNeuron.UpdateNeuron()

        expected = x1 * w1 + x2 * w2 + 0 * w3 + b + oldOut * rw
        self.assertEqual(expected, testNeuron.neuron['Output'])

    def test_NeuralNetwork_CreateInitialGraph(self):
        self.assertEqual(True, True)

    def test_NeuralNetwork_UpdateGraph(self):
        self.assertEqual(True, True)

    def test_NeuralNetwork_FindNeuronsInLayer(self):
        self.assertEqual(True, True)

    def test_NeuralNetwork_AddNeuron(self):
        self.assertEqual(True, True)

    def test_NeuralNetwork_DrawGraph(self):
        self.assertEqual(True, True)

    def test_NeuralNetwork_InternalDrawGraph(self):
        self.assertEqual(True, True)

    def test_NeuralNetwork_LogGraph(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
