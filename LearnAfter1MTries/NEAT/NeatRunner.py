from LearnAfter1MTries.NEAT.GeneticAlgorithm import GA
from LearnAfter1MTries.NEAT.NeuralNetwork import NeuralNetwork
from LearnAfter1MTries.NEAT.GeneticAlgorithm import *
import itertools


class Neat:
    def __init__(self, numInputs, numOutputs,
                 PopSize=100, CrossOverRate=0.7, MutationRate=0.001, ChromoLength=70, GeneLength=2,
                 selectionType='weightedRouletteWheel', crossOverType='randomSwapPoint', mutateType='Binary',
                 chromoType='Binary',
                 stringMutate='swap', DecodeDict=[], numGeneration=100, fitnessTest=[],
                 infoBoard=[], progressGen=[], progressOverall=[]):
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.PopSize = PopSize
        self.CrossOverRate = CrossOverRate
        self.MutationRate = MutationRate
        self.GeneLength = GeneLength
        self.selectionType = selectionType
        self.crossOverType = crossOverType
        self.mutateType = mutateType
        self.chromoType = chromoType
        self.stringMutate = stringMutate
        self.DecodeDict = DecodeDict
        self.numGeneration = numGeneration
        self.fitnessTest = fitnessTest
        self.infoBoard = infoBoard
        self.progressGen = progressGen
        self.progressOverall = progressOverall

        self.GA = []
        self.NeatListGenomes = []

    def CreatePopulation(self):
        for num in range(0, self.PopSize):
            net = NeuralNetwork(self.numInputs, self.numOutputs)
            net.CreateInitialGraph()
            self.NeatListGenomes.append(net)

        self.GA = GA(PopSize=self.PopSize,
                     CrossOverRate=self.CrossOverRate,
                     MutationRate=self.MutationRate,
                     ChromoLength=self.MutationRate,
                     GeneLength=self.GeneLength,
                     selectionType=self.selectionType,
                     crossOverType=self.crossOverType, mutateType=self.mutateType,
                     chromoType=self.chromoType,
                     stringMutate=self.stringMutate,
                     DecodeDict=self.DecodeDict,
                     numGeneration=self.numGeneration,
                     fitnessTest=self.fitnessTest,
                     infoBoard=self.infoBoard,
                     progressGen=self.progressGen,
                     progressOverall=self.progressOverall)

    def GenerateBabies(self, mum, dad):
        maxMumNode = -1
        maxDadNode = -1
        for mumNode in mum:
            if mumNode > maxMumNode:
                maxMumNode = mumNode

            for dadNode in dad:
                if dadNode > maxDadNode:
                    maxDadNode = dadNode

                if mumNode['Layer'] == dadNode['Layer']:
                    if mumNode['NodeNum'] == dadNode['NodeNum']:
                        baby1, baby2 = self.CrossOver(mum, dad)
                        baby1 = self.Mutate(baby1)
                        baby2 = self.Mutate(baby2)

        for mumNode in mum:
            if mumNode > maxDadNode:
                self.Mutate(mumNode)
                self.AddNodeOrNotAdd(mumNode, dad)

        for dadNode in dad:
            if dadNode > maxMumNode:
                self.Mutate(dadNode)
                self.AddNodeOrNotAdd(dadNode, mum)




if __name__ == '__main__':
    print("NeatRunner")
    test = Neat(numInputs=2, numOutputs=1, PopSize=2)
    test.CreatePopulation()

    # Solving XOR
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    y = [0, 1, 1, 0]


    def TestFitness(NN, X, y):
        totalError = 0
        for index in range(0, len(X)):
            NN.UpdateGraph(X[index])
            totalError += math.fabs(y[index] - NN.NeuralNet['Output'])
        NN.NeuralNet['Fitness'] = 1 / (1 + totalError)


    print(test.NeatListGenomes[0])
    print(test.NeatListGenomes[1])

    print("--Test Fitness----")
    print(TestFitness(test.NeatListGenomes[0], X, y))
    print(TestFitness(test.NeatListGenomes[1], X, y))
    print('---Update Fitness---')
    print(test.NeatListGenomes[0].NeuralNet['Fitness'])
    print(test.NeatListGenomes[1].NeuralNet['Fitness'])

    test.GenerateBabies(test.NeatListGenomes[0].NeuralNet['Network'], test.NeatListGenomes[0].NeuralNet['Network'])

    addedNeuron = test.NeatListGenomes[0].NeuralNet['Network'][0]
    print(addedNeuron)

    test.NeatListGenomes[0].AddNeuron(addedNeuron)
    print(test.NeatListGenomes[0])

    test.GenerateBabies(test.NeatListGenomes[0].NeuralNet['Network'], test.NeatListGenomes[1].NeuralNet['Network'])
