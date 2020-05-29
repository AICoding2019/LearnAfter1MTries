import unittest
from LearnAfter1MTries.TWEANN.GeneticAlgorithm import *
import statistics as stat


class GeneticAlgorithmTestCase(unittest.TestCase):
    def test_BinToInt(self):
        actual = BinToInt([1, 1, 1])
        expected = 7
        self.assertEqual(actual, expected)

    def test_PerturbRealNumber_Mean(self):
        data = [PerturbRealNumber(0, 1) for _ in range(0, int(1E4))]
        dataMean = stat.mean(data)

        if abs(dataMean) < 1E-2:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)

    def test_PerturbRealNumber_Std(self):
        data = [PerturbRealNumber(0, 1) for _ in range(0, int(1E4))]
        dataStd = stat.stdev(data)

        if dataStd > 0.5:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)

    def test_CrossOverNeuralNetwork_MumEqualDad(self):
        testGA = GA()
        mumNeuralNet = NeuralNetwork(2, 2)
        dadNeuralNet = NeuralNetwork(2, 2)
        mumNeuralNet.NeuralNet = {
            'Network': [{'ID': 0, 'Weights': [0, 0], 'Bias': 10, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 0,
                         'Layer': 0, 'Activation': 'relu', 'NodeNum': 0, 'Inputs': [], 'Output': 0},
                        {'ID': 1, 'Weights': [1, 1], 'Bias': 11, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 0,
                         'Layer': 0, 'Activation': 'relu', 'NodeNum': 1, 'Inputs': [], 'Output': 0},
                        {'ID': 2, 'Weights': [2, 2], 'Bias': 12, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 0,
                         'Layer': 1, 'Activation': 'relu', 'NodeNum': 0, 'Inputs': [], 'Output': 0},
                        {'ID': 3, 'Weights': [3, 3], 'Bias': 13, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 0,
                         'Layer': 1, 'Activation': 'relu', 'NodeNum': 1, 'Inputs': [], 'Output': 0},
                        {'ID': 4, 'Weights': [4, 4], 'Bias': 14, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 0,
                         'Layer': 2, 'Activation': 'relu', 'NodeNum': 0, 'Inputs': [], 'Output': 0},
                        {'ID': 5, 'Weights': [5, 5], 'Bias': 15, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 0,
                         'Layer': 2, 'Activation': 'relu', 'NodeNum': 1, 'Inputs': [], 'Output': 0},
                        {'ID': 6, 'Weights': [6, 6], 'Bias': 16, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 0,
                         'Layer': -1, 'Activation': 'relu', 'NodeNum': 0, 'Inputs': [], 'Output': 0},
                        {'ID': 7, 'Weights': [7, 7], 'Bias': 17, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 0,
                         'Layer': -1, 'Activation': 'relu', 'NodeNum': 1,
                         'Inputs': [], 'Output': 0}],
            'Output': [],
            'Layers': 2,
            'Species': [2, 2, 2, 2, 2],  # [inputs, hiddenlayers, outputs]
            'MaxNodes': 2,
            'Fitness': None
        }
        dadNeuralNet.NeuralNet = {
            'Network': [{'ID': 0, 'Weights': [0, 0], 'Bias': 0, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 1,
                         'Layer': 0, 'Activation': 'sigmoid', 'NodeNum': 0, 'Inputs': [], 'Output': 0},
                        {'ID': 1, 'Weights': [1, 1], 'Bias': 1, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 1,
                         'Layer': 0, 'Activation': 'sigmoid', 'NodeNum': 1, 'Inputs': [], 'Output': 0},
                        {'ID': 2, 'Weights': [2, 2], 'Bias': 2, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 1,
                         'Layer': 1, 'Activation': 'sigmoid', 'NodeNum': 0, 'Inputs': [], 'Output': 0},
                        {'ID': 3, 'Weights': [3, 3], 'Bias': 3, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 1,
                         'Layer': 1, 'Activation': 'sigmoid', 'NodeNum': 1, 'Inputs': [], 'Output': 0},
                        {'ID': 4, 'Weights': [4, 4], 'Bias': 4, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 1,
                         'Layer': 2, 'Activation': 'sigmoid', 'NodeNum': 0, 'Inputs': [], 'Output': 0},
                        {'ID': 5, 'Weights': [5, 5], 'Bias': 5, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 1,
                         'Layer': 2, 'Activation': 'sigmoid', 'NodeNum': 1, 'Inputs': [], 'Output': 0},
                        {'ID': 6, 'Weights': [6, 6], 'Bias': 6, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 1,
                         'Layer': -1, 'Activation': 'sigmoid', 'NodeNum': 0, 'Inputs': [], 'Output': 0},
                        {'ID': 7, 'Weights': [7, 7], 'Bias': 7, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 1,
                         'Layer': -1, 'Activation': 'sigmoid', 'NodeNum': 1,
                         'Inputs': [], 'Output': 0}],
            'Output': [],
            'Layers': 2,
            'Species': [2, 2, 2, 2, 2],  # [inputs, hiddenlayers, outputs]
            'MaxNodes': 2,
            'Fitness': None
        }

        mumGenome = {'chromo': mumNeuralNet, 'Fitness': 0, 'Info': []}
        dadGenome = {'chromo': dadNeuralNet, 'Fitness': 0, 'Info': []}

        baby1, baby2 = testGA.CrossOverNeuralNetwork(mumGenome, dadGenome)
        if baby1['chromo'].NeuralNet['Species'] == mumNeuralNet['Species'] and baby1['chromo'].NeuralNet['Species'] == mumNeuralNet['Species']:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)

    def test_CrossOverNeuralNetwork_MumNotEqualDad(self):
        testGA = GA()
        mumNeuralNet = NeuralNetwork(2, 2)
        dadNeuralNet = NeuralNetwork(2, 2)
        mumNeuralNet.NeuralNet = {
            'Network': [{'ID': 0, 'Weights': [0, 0], 'Bias': 10, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 0,
                         'Layer': 0, 'Activation': 'relu', 'NodeNum': 0, 'Inputs': [], 'Output': 0},
                        {'ID': 1, 'Weights': [1, 1], 'Bias': 11, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 0,
                         'Layer': 0, 'Activation': 'relu', 'NodeNum': 1, 'Inputs': [], 'Output': 0},
                        {'ID': 2, 'Weights': [2, 2], 'Bias': 12, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 0,
                         'Layer': 1, 'Activation': 'relu', 'NodeNum': 0, 'Inputs': [], 'Output': 0},
                        {'ID': 3, 'Weights': [3, 3], 'Bias': 13, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 0,
                         'Layer': 1, 'Activation': 'relu', 'NodeNum': 1, 'Inputs': [], 'Output': 0},
                        {'ID': 4, 'Weights': [4, 4], 'Bias': 14, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 0,
                         'Layer': 2, 'Activation': 'relu', 'NodeNum': 0, 'Inputs': [], 'Output': 0},
                        {'ID': 5, 'Weights': [5, 5], 'Bias': 15, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 0,
                         'Layer': 2, 'Activation': 'relu', 'NodeNum': 1, 'Inputs': [], 'Output': 0},
                        {'ID': 6, 'Weights': [6, 6], 'Bias': 16, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 0,
                         'Layer': -1, 'Activation': 'relu', 'NodeNum': 0, 'Inputs': [], 'Output': 0},
                        {'ID': 7, 'Weights': [7, 7], 'Bias': 17, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 0,
                         'Layer': -1, 'Activation': 'relu', 'NodeNum': 1,
                         'Inputs': [], 'Output': 0}],
            'Output': [],
            'Layers': 4,
            'Species': [2, 2, 2, 2, 2],  # [inputs, hiddenlayers, outputs]
            'MaxNodes': 2,
            'Fitness': None
        }
        dadNeuralNet.NeuralNet = {
            'Network': [{'ID': 100, 'Weights': [0, 0], 'Bias': 0, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 1,
                         'Layer': 0, 'Activation': 'sigmoid', 'NodeNum': 0, 'Inputs': [], 'Output': 0},
                        {'ID': 200, 'Weights': [1, 1], 'Bias': 1, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 1,
                         'Layer': 0, 'Activation': 'sigmoid', 'NodeNum': 1, 'Inputs': [], 'Output': 0},
                        {'ID': 300, 'Weights': [6, 6], 'Bias': 6, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 1,
                         'Layer': -1, 'Activation': 'sigmoid', 'NodeNum': 0, 'Inputs': [], 'Output': 0},
                        {'ID': 400, 'Weights': [7, 7], 'Bias': 7, 'Enabled': 1, 'Recurrent': 0, 'RecurrentWeight': 1,
                         'Layer': -1, 'Activation': 'sigmoid', 'NodeNum': 1,
                         'Inputs': [], 'Output': 0}],
            'Output': [],
            'Layers': 2,
            'Species': [2, 2, 2],  # [inputs, hiddenlayers, outputs]
            'MaxNodes': 2,
            'Fitness': None
        }

        mumGenome = {'chromo': mumNeuralNet, 'Fitness': 0, 'Info': []}
        dadGenome = {'chromo': dadNeuralNet, 'Fitness': 0, 'Info': []}

        baby1, baby2 = testGA.CrossOverNeuralNetwork(mumGenome, dadGenome)
        if baby1['chromo'].NeuralNet['Species'] == mumNeuralNet['Species'] and baby1['chromo'].NeuralNet['Species'] == \
                mumNeuralNet['Species']:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)



if __name__ == '__main__':
    unittest.main()
