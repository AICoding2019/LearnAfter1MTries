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


if __name__ == '__main__':
    unittest.main()
