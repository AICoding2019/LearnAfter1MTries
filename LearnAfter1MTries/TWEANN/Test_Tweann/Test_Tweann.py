import unittest
import json
import os
from LearnAfter1MTries.TWEANN.TweannRunner import *


class TweannTestCase(unittest.TestCase):
    def test_MutateByAddingNode(self):
        numInputs = 16000
        numOutputs = 30
        testTweann = Tweann(numInputs, numOutputs, PopSize=500, numGeneration=100000, addNodeMutationRate=0.1,
                            selectionType='rouletteWheel', displayer=False, dirStore='UnitTest')
        NNtoMutate = NeuralNetwork(numInputs, numOutputs)
        NNtoMutate.CreateInitialGraph()
        for _ in range(0, 30):
            NNtoMutate = testTweann.MutateByAddingNode(NNtoMutate)

        if sum(NNtoMutate.NeuralNet['Species'][1:-1]) == 30:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)

    def test_CrossOver(self):
        numInputs = 16000
        numOutputs = 30
        testTweann = Tweann(numInputs, numOutputs, PopSize=500, numGeneration=100000, addNodeMutationRate=0.1,
                            selectionType='rouletteWheel', displayer=False, dirStore='UnitTest')
        NNtoCross = NeuralNetwork(numInputs, numOutputs)
        NNtoCross.CreateInitialGraph()
        for _ in range(0, 30):
            baby1, baby2 = testTweann.CrossOver(NNtoCross, NNtoCross)

        self.assertEqual(True, True)

    def test_TweannRunnerList(self):
        def TestFitness(NN):
            # Solving XOR
            X = [[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]]
            y = [0, 1, 1, 0]
            totalError = 0
            numberCorrect = 0
            prediction = []
            predictionBin = []
            for index in range(0, len(X)):
                NN.UpdateGraph(X[index])
                totalError += fabs(y[index] - NN.NeuralNet['Output'][0])
                if NN.NeuralNet['Output'][0] < 0.5:
                    predict = 0
                else:
                    predict = 1

                if y[index] == predict:
                    numberCorrect += 1

                prediction.append(NN.NeuralNet['Output'][0])
                predictionBin.append(predict)
                NN.NeuralNet['Output'] = []
            fitness = (numberCorrect / 4 + (1 / (1 + totalError))) / 2
            # NN.NeuralNet['Fitness'] = fitness

            return fitness, [numberCorrect, NN.NeuralNet['Species'], prediction, predictionBin]

        testXOR = Tweann(numInputs=2, numOutputs=1, PopSize=100, numGeneration=1000, selectionType='rouletteWheel',
                         fitnessTestFunction=TestFitness)
        testXOR.Evolve()
        self.assertEqual(True, True)

    def test_TweannRunnerFolder(self):
        def TestFitness(NN):
            # Solving XOR
            X = [[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]]
            y = [0, 1, 1, 0]
            totalError = 0
            numberCorrect = 0
            prediction = []
            predictionBin = []
            for index in range(0, len(X)):
                NN.UpdateGraph(X[index])
                totalError += fabs(y[index] - NN.NeuralNet['Output'][0])
                if NN.NeuralNet['Output'][0] < 0.5:
                    predict = 0
                else:
                    predict = 1

                if y[index] == predict:
                    numberCorrect += 1

                prediction.append(NN.NeuralNet['Output'][0])
                predictionBin.append(predict)
                NN.NeuralNet['Output'] = []
            fitness = (numberCorrect / 4 + (1 / (1 + totalError))) / 2
            # NN.NeuralNet['Fitness'] = fitness

            return fitness, [numberCorrect, NN.NeuralNet['Species'], prediction, predictionBin]

        testXOR = Tweann(numInputs=2, numOutputs=1, PopSize=100, numGeneration=2, addNodeMutationRate=0.5,
                         selectionType='rouletteWheel',displayer=False,
                         fitnessTestFunction=TestFitness, dirStore='UnitTest')
        testXOR.Evolve()
        self.assertEqual(True, True)

    def test_WriteToFolder(self):
        testWriteToFolder = Tweann(numInputs=2, numOutputs=1, dirStore='Test')
        expected = {
            'Network': [],
            'Output': [],
            'Layers': 0,
            'Species': [2, 0, 1],  # [inputs, hiddenlayers, outputs]
            'MaxNodes': 1,
            'Fitness': 1
        }
        num = 0
        testWriteToFolder.WriteToFolder(expected, num)

        with open('Test/Child0.json') as f:
            actual = json.load(f)

        self.assertEqual(expected, actual)

    def test_CreatePopulationFileStore(self):
        dir = 'Test'
        testCreatePopFileStore = Tweann(numInputs=16000, numOutputs=30, PopSize=500, dirStore=dir)
        testCreatePopFileStore.CreatePopulationFileStore()

        self.assertEqual(True, True)

    def test_ReadFromFolder(self):
        dir = 'Test'
        testCreatePopFileStore = Tweann(numInputs=2, numOutputs=1, PopSize=2, dirStore=dir)
        testCreatePopFileStore.CreatePopulationFileStore()
        actual = os.listdir(dir)

        expected = ['Child0.json', 'Child1.json']

        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
