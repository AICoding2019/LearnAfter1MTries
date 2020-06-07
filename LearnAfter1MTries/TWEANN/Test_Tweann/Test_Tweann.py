import unittest
from LearnAfter1MTries.TWEANN.TweannRunner import *


class TweannTestCase(unittest.TestCase):
    def test_MutateByAddingNode(self):
        numInputs = 2
        numOutputs = 1
        testTweann = Tweann(numInputs, numOutputs)
        NNtoMutate = NeuralNetwork(numInputs, numOutputs)
        NNtoMutate.CreateInitialGraph()
        for _ in range(0,30):
            NNtoMutate=testTweann.MutateByAddingNode(NNtoMutate)

        if sum(NNtoMutate.NeuralNet['Species'][1:-1]) == 30:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)

    def test_TweannRunner(self):
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

        testXOR = Tweann(numInputs=2, numOutputs=1, PopSize=1000, numGeneration=60, selectionType='rouletteWheel',
                         fitnessTestFunction=TestFitness)
        testXOR.Evolve()
        self.assertEqual(True, True)



if __name__ == '__main__':
    unittest.main()
