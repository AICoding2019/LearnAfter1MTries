# from LearnAfter1MTries.TWEANN.GeneticAlgorithm import GA
# from LearnAfter1MTries.TWEANN.NeuralNetwork import NeuralNetwork, Neuron
from LearnAfter1MTries.TWEANN.GeneticAlgorithm import *
from random import random
from random import choice
from random import randrange
from math import fabs
import numpy as np
from copy import deepcopy
from os import mkdir, path
from shutil import rmtree
import json
import rapidjson
import orjson


class Tweann:
    def __init__(self, numInputs, numOutputs, PopSize=100, CrossOverRate=0.7, MutationRate=0.1,
                 addNodeMutationRate=0.1, selectionType='weightedRouletteWheel', DecodeDict=[],
                 numGeneration=2, fitnessTestFunction=[], displayer=True, dirStore=[]):
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.PopSize = PopSize
        self.CrossOverRate = CrossOverRate
        self.MutationRate = MutationRate
        self.selectionType = selectionType
        self.DecodeDict = DecodeDict
        self.numGeneration = numGeneration
        self.fitnessTest = fitnessTestFunction
        self.AddNodeMutationRate = addNodeMutationRate
        self.displayer = displayer
        self.dirStore = dirStore

        self.GA = GA(PopSize=self.PopSize,
                     CrossOverRate=self.CrossOverRate,
                     MutationRate=self.MutationRate,
                     selectionType=self.selectionType,
                     DecodeDict=self.DecodeDict,
                     numGeneration=self.numGeneration,
                     fitnessTestFunction=self.fitnessTest,
                     genomeKey='Network',
                     chromoType='Network',
                     mutationCustomFunction=self.Mutate,
                     crossOverCustomFunction=self.CrossOver,
                     setPopFlag=True,
                     stringMutate='swap',
                     progressGen=self.ProgressDisplayer,
                     progressOverall=self.ProgressOverallDisplayer,
                     infoBoard=self.InfoBoard,
                     dirStore=self.dirStore
                     )
        self.NeatListGenomes = []

        self.makeStorageFolder()

    def makeStorageFolder(self):
        if len(self.dirStore) != 0:
            if path.exists(self.dirStore):
                rmtree(self.dirStore)

            mkdir(self.dirStore)

    def CreatePopulation(self):
        if len(self.dirStore) == 0:
            self.CreatePopulationList()
        else:
            self.CreatePopulationFileStore()

    def CreatePopulationList(self):
        for num in range(0, self.PopSize + 2):
            net = NeuralNetwork(self.numInputs, self.numOutputs)
            net.CreateInitialGraph()
            self.NeatListGenomes.append(net)

        self.GA.setStartPopulation(self.NeatListGenomes)
        self.NeatListGenomes = []

    def CreatePopulationFileStore(self):
        for num in range(0, self.PopSize + 2):
            net = NeuralNetwork(self.numInputs, self.numOutputs)
            net.CreateInitialGraph()
            child = {'chromo': net.NeuralNet,
                     'Fitness': 0,
                     'Info': [],
                     'numInputs': self.numInputs,
                     'numOutputs': self.numOutputs
                     }
            self.WriteToFolder(child, num)
            print(f"Creating Child{num}")

    def WriteToFolderJson(self, data, num):
        fileName = 'Child' + str(num)
        with open(self.dirStore + '/' + fileName + '.json', 'w') as json_file:
            dump(data, json_file) #, indent=4

    def WriteToFolderTxt(self, data, num):
        fileName = 'Child' + str(num)
        with open(self.dirStore + '/' + fileName + '.txt', 'w') as txt_file:
            txt_file.write(str(data))

    def WriteToFolder(self, data, num):
        fileName = 'Child' + str(num)
        writeData = orjson.dumps(data, option=orjson.OPT_NAIVE_UTC)
        with open(self.dirStore + '/' + fileName + '.json', 'wb') as json_file:
            json_file.write(writeData)
            #dump(writeData, json_file)  #, indent=4

    def ReadFromFolderJson(self, num):
        fileName = 'Child' + str(num)
        with open(self.dirStore + '/' + fileName + '.json', 'w') as json_file:
            data = json.load(json_file)
        return data

    def ReadFromFolder(self, num):
        fileName = 'Child' + str(num)
        with open(self.dirStore + '/' + fileName + '.json', 'w') as json_file:
            data = orjson.loads(json_file)
        return data

    def GenerateBabies(self, mum, dad):
        maxMumNodeNum = -1
        maxDadNodeNum = -1
        layers = 0
        baby1 = NeuralNetwork(self.numInputs, self.numOutputs)
        baby2 = NeuralNetwork(self.numInputs, self.numOutputs)

        if mum['Fitness'] > dad['Fitness']:
            fittest = 'mum'
        else:
            fittest = 'dad'

        for mumNeuron in mum:
            if mumNeuron['NodeNum'] > maxMumNodeNum:
                maxMumNodeNum = mumNeuron['NodeNum']

            for dadNeuron in dad:
                if dadNeuron['NodeNum'] > maxDadNodeNum:
                    maxDadNodeNum = dadNeuron['NodeNum']

                if mumNeuron['Layer'] == dadNeuron['Layer']:
                    if mumNeuron['NodeNum'] == dadNeuron['NodeNum']:
                        baby1Neuron, baby2Neuron = self.CrossOver(mumNeuron, dadNeuron, fittest)
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
                self.AddNodeOrNot(neuron, baby2.NeuralNet['Network'])

        if random() < self.AddNodeMutationRate:
            self.MutateByAddingNode(baby1.NeuralNet)
        if random() < self.AddNodeMutationRate:
            self.MutateByAddingNode(baby2.NeuralNet)

        return baby1, baby2

    def Mutate(self, NN):
        mutatedNN = self.MutateOneNeuronElement(NN)
        return mutatedNN

    def MutateOneNeuronElement(self, NN):
        mutatedNN = NN
        selectedNeuron = randrange(0, len(mutatedNN.NeuralNet['Network']))
        neuron = mutatedNN.NeuralNet['Network'][selectedNeuron]
        selectedWeight = randrange(0, len(neuron['Weights']))

        neuron['Weights'][selectedWeight] = (np.array(neuron['Weights'][selectedWeight]) +
                             (-self.MutationRate + 2 * self.MutationRate * random())).tolist()
        neuron['Bias'] = (np.array(neuron['Bias']) +
                          (-self.MutationRate + 2 * self.MutationRate * random())).tolist()
        neuron['RecurrentWeight'] = (np.array(neuron['RecurrentWeight']) +
                                     (-self.MutationRate + 2 * self.MutationRate * random())).tolist()

        if random() < self.MutationRate:
            neuron['Recurrent'] = choice([0, 1])

        if random() < self.MutationRate:
            neuron['Activation'] = choice(Neuron().activationFunc)

        if random() < self.MutationRate:
            neuron['Enabled'] = choice([0, 1])

        if random() < self.AddNodeMutationRate:
            mutatedNN = self.MutateByAddingNode(mutatedNN)

        return mutatedNN

    def MutateAllNeurons(self, NN):
        mutatedNN = NN
        for neuron in mutatedNN.NeuralNet['Network']:
            neuron['Weights'] = (np.array(neuron['Weights']) +
                                 (-self.MutationRate + 2 * self.MutationRate * random())).tolist()
            neuron['Bias'] = (np.array(neuron['Bias']) +
                              (-self.MutationRate + 2 * self.MutationRate * random())).tolist()
            neuron['RecurrentWeight'] = (np.array(neuron['RecurrentWeight']) +
                                         (-self.MutationRate + 2 * self.MutationRate * random())).tolist()

            if random() < self.MutationRate:
                neuron['Recurrent'] = choice([0, 1])

            if random() < self.MutationRate:
                neuron['Activation'] = choice(Neuron().activationFunc)

            if random() < self.MutationRate:
                neuron['Enabled'] = choice([0, 1])

        if random() < self.AddNodeMutationRate:
            mutatedNN = self.MutateByAddingNode(mutatedNN)

        return mutatedNN

    def CrossOverRealNumbers(self, mumGene, dadGene, fittest):
        Max = max(mumGene, dadGene)
        Min = min(mumGene, dadGene)

        if fittest == 'mum':
            baby1 = mumGene + self.CrossOverRate * (Max - Min) * random()
            baby2 = mumGene + self.CrossOverRate * (Max - Min) * random()
        else:

            baby1 = dadGene + self.CrossOverRate * (Max - Min) * random()
            baby2 = dadGene + self.CrossOverRate * (Max - Min) * random()

        return baby1, baby2

    def CrossOverOther(self, mumGene, dadGene, fittest):
        if random() < self.CrossOverRate:
            if fittest == 'mum':
                baby1 = mumGene
                baby2 = mumGene
            else:
                baby1 = dadGene
                baby2 = dadGene
        else:
            baby1 = mumGene
            baby2 = dadGene

        return baby1, baby2

    def CrossOver(self, mumNeuron, dadNeuron):
        baby1 = Neuron()
        baby2 = Neuron()

        baby1.neuron['Weight'], baby2.neuron['Weight'] = self.CrossOverRealNumbers(mumNeuron['Weight'],
                                                                                   dadNeuron['Weight'])
        baby1.neuron['Bias'], baby2.neuron['Bias'] = self.CrossOverRealNumbers(mumNeuron['Bias'], dadNeuron['Bias'])
        baby1.neuron['RecurrentWeight'], baby2.neuron['RecurrentWeight'] = \
            self.CrossOverRealNumbers(mumNeuron['RecurrentWeight'], dadNeuron['RecurrentWeight'])

        baby1.neuron['Activation'], baby1.neuron['Activation'] = self.CrossOverOther(mumNeuron['Activation'],
                                                                                     dadNeuron['Activation'])

        baby1.neuron['Recurrent'], baby1.neuron['Recurrent'] = self.CrossOverOther(mumNeuron['Recurrent'],
                                                                                   dadNeuron['Recurrent'])

        baby1.neuron['Enabled'], baby1.neuron['Enabled'] = self.CrossOverOther(mumNeuron['Enabled'],
                                                                               dadNeuron['Enabled'])

        return baby1, baby2

    def ProgressDisplayer(self, info, fittestEver=True):
        if self.displayer:
            self.GA.FittestGenome['chromo'].InternalDrawGraph()
        else:
            pass

    # print(f"----{self.GA.FittestGenomeEver['chromo'].NeuralNet['Species']}")

    def ProgressOverallDisplayer(self, info, fittestEver=True):
        pass

    def InfoBoard(self, text):
        print(text)
        pass

    def Epoch(self):
        self.GA.Epoch()

    def Evolve(self):
        print("Creating Population")
        self.CreatePopulation()
        print("Population Created")
        self.Epoch()

    @staticmethod
    def AddNodeOrNot(neuron, network):
        if random() < 0.5:
            network.AddNeuron(neuron)

    # @staticmethod
    def MutateByAddingNode(self, NNtoMutate):
        if len(self.dirStore) == 0:
            NN = deepcopy(NNtoMutate)
        else:
            NN = NNtoMutate

        if NN.NeuralNet['Species'][1] == 0:
            layerToAddNeuron = 0
        else:
            layerToAddNeuron = randrange(0, NN.NeuralNet['Layers'] + 2)

        maxNodesInLayer = -1
        maxNodes = 0

        for neuron in NN.NeuralNet['Network']:

            if maxNodes < neuron['NodeNum']:
                maxNodes = neuron['NodeNum']
            if neuron['Layer'] == layerToAddNeuron:
                if maxNodesInLayer < neuron['NodeNum']:
                    maxNodesInLayer = neuron['NodeNum']

        addNeuron = Neuron()
        addNeuron.neuron['ID'] = maxNodes + 1
        addNeuron.neuron['Weights'] = [random() for _ in range(0, len(neuron['Weights']))]
        addNeuron.neuron['Bias'] = random()
        addNeuron.neuron['Enabled'] = 1
        addNeuron.neuron['Recurrent'] = choice([0, 1])
        addNeuron.neuron['RecurrentWeight'] = random()
        addNeuron.neuron['Layer'] = layerToAddNeuron
        addNeuron.neuron['Activation'] = choice(addNeuron.activationFunc)
        addNeuron.neuron['Inputs'] = neuron['Inputs']
        addNeuron.neuron['NodeNum'] = maxNodesInLayer + 1
        addNeuron.neuron['Output'] = 0

        NN.AddNeuron(addNeuron.neuron)
        return NN


if __name__ == '__main__':
    print("TweannRunner")


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

    # while True:
    #    pass
