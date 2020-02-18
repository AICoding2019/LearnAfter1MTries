from LearnAfter1MTries.TWEANN.GeneticAlgorithm import GA
from LearnAfter1MTries.TWEANN.NeuralNetwork import NeuralNetwork, Neuron
from LearnAfter1MTries.TWEANN.GeneticAlgorithm import *
from random import random
from random import choice


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
        maxMumNodeNum = -1
        maxDadNodeNum = -1
        layers = 0
        baby1 = NeuralNetwork(self.numInputs, self.numOutputs)
        baby2 = NeuralNetwork(self.numInputs, self.numOutputs)

        for mumNeuron in mum:
            if mumNeuron['NodeNum'] > maxMumNodeNum:
                maxMumNodeNum = mumNeuron['NodeNum']

            for dadNeuron in dad:
                if dadNeuron['NodeNum'] > maxDadNodeNum:
                    maxDadNodeNum = dadNeuron['NodeNum']

                if mumNeuron['Layer'] == dadNeuron['Layer']:
                    if mumNeuron['NodeNum'] == dadNeuron['NodeNum']:
                        baby1Neuron, baby2Neuron = self.CrossOver(mumNeuron, dadNeuron)
                        baby1Neuron = self.Mutate(baby1Neuron)
                        baby2Neuron = self.Mutate(baby2Neuron)
                        baby1.NeuralNet['Network'].append(baby1Neuron.copy())
                        baby2.NeuralNet['Network'].append(baby2Neuron.copy())

                if mumNeuron['Layer'] > layers:
                    layers = mumNeuron['Layer']

                if dadNeuron['Layer'] > layers:
                    layers = dadNeuron['Layer']

        for mumNeuron in mum:
            if mumNeuron['NodeNum'] > maxDadNodeNum:
                neuron = self.Mutate(mumNeuron)
                self.AddNodeOrNot(neuron, baby1.NeuralNet['Network'])

        for dadNeuron in dad:
            if dadNeuron['NodeNum'] > maxMumNodeNum:
                neuron = self.Mutate(dadNeuron)
                self.AddNodeOrNot(neuron, baby1.NeuralNet['Network'])

        baby1 = self.MutateByAddingNode(baby1, baby1layers, baby1MaxNum)
        baby2 = self.MutateByAddingNode(baby2, baby2layers, baby2MaxNum)

        return baby1, baby2

    def Mutate(self, neuron):
        return neuron

    def CrossOver(self, mumNeuron, dadNeuron):
        return mumNeuron.copy(), dadNeuron.copy()

    @staticmethod
    def AddNodeOrNot(neuron, network):
        if random() < 0.5:
            network.AddNeuron(neuron)

    @staticmethod
    def MutateByAddingNode(baby, babyLayers, babyMaxNum):
        addNeuron = Neuron()
        addNeuron.neuron['ID'] = [],
        addNeuron.neuron['Weights'] = random(),
        addNeuron.neuron['Bias'] = random(),
        addNeuron.neuron['Enabled'] = 1,
        addNeuron.neuron['Recurrent'] = choice([0, 1]),
        addNeuron.neuron['RecurrentWeight'] = random(),
        addNeuron.neuron['Layer'] = random.randrange(babyLayers+2),
        addNeuron.neuron['Activation'] = choice(addNeuron.activationFunc),
        addNeuron.neuron['NodeNum'] = 0,
        addNeuron.neuron['Inputs'] = [],
        addNeuron.neuron['Output'] = 0,
        baby.AddNeuron(addNeuron)

        return baby


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
