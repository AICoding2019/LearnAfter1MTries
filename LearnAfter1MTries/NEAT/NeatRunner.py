from LearnAfter1MTries.NEAT.GeneticAlgorithm import GA
from LearnAfter1MTries.NEAT.NeuralNetwork import NeuralNetwork


class Neat:
    def __init__(self, PopSize=100, CrossOverRate=0.7, MutationRate=0.001, ChromoLength=70, GeneLength=2,
                 selectionType='weightedRouletteWheel', crossOverType='randomSwapPoint', mutateType='Binary',
                 chromoType='Binary',
                 stringMutate='swap', DecodeDict=[], numGeneration=100, fitnessTest=[],
                 infoBoard=[], progressGen=[], progressOverall=[]):
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
            self.NeatListGenomes.append(net.copy())

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
