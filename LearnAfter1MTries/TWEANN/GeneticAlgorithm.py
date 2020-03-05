import random as rd
from random import random
from random import randint
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
from LearnAfter1MTries.TWEANN.NeuralNetwork import *


def BinToInt(binary):
    val = 0
    multiplier = 1

    for bit in reversed(binary):
        val += bit * multiplier
        multiplier *= 2

    return val


def PerturbRealNumber(realNumber, perturbationRate):
    if random() < perturbationRate:
        perturbation = realNumber + (-perturbationRate + 2 * perturbationRate * random())
    else:
        perturbation = realNumber

    return perturbation


def FlipBoolean(boolean, FlipRate):
    if random() < FlipRate:
        if boolean == 1:
            flip = 0
        else:
            flip = 1
    else:
        flip = boolean

    return flip


def MutateChromeRandomResetting(chromo, mutationRate, chromoRange):
    """"""
    '''https: // www.geeksforgeeks.org / mutation - algorithms -
    ''for -string - manipulation - ga /
    '''''''''
    mutation = chromo.copy()
    randomIndex = rd.randrange(len(chromo))

    if random() < mutationRate:
        mutation[randomIndex] = random.choice(chromoRange)

    return mutation


def CrossOverPartiallyMapped(Mum, Dad, chromoLength, crossOverRate=0.7):
    mum = Mum.copy()
    dad = Dad.copy()
    randomIndex1 = rd.randrange(chromoLength)
    randomIndex2 = rd.randrange(chromoLength)

    if randomIndex1 < randomIndex2:
        beginning = randomIndex1
        end = randomIndex2
    else:
        beginning = randomIndex2
        end = randomIndex1

    mumSwap = mum['chromo'][beginning:end]
    dadSwap = dad['chromo'][beginning:end]

    mumMap0 = 0
    mumMap1 = 0

    dadMap0 = 0
    dadMap1 = 0
    # find the index of elements needed to swap
    for mumSwapElement, dadSwapElement in zip(mumSwap, dadSwap):
        for index, (mumElement, dadElement) in enumerate(zip(mum['chromo'], dad['chromo'])):
            if mumElement == mumSwapElement:
                mumMap0 = index
            if mumElement == dadSwapElement:
                mumMap1 = index

            if dadElement == mumSwapElement:
                dadMap0 = index
            if dadElement == dadSwapElement:
                dadMap1 = index
        mum['chromo'][mumMap0], mum['chromo'][mumMap1] = mum['chromo'][mumMap1], mum['chromo'][mumMap0]
        dad['chromo'][dadMap0], dad['chromo'][dadMap1] = dad['chromo'][dadMap1], dad['chromo'][dadMap0]

    baby1 = mum.copy()
    baby2 = dad.copy()
    return baby1, baby2


def CrossOverOrder(mum, dad, crossOverRate, chromoLength):
    baby1 = mum.copy()
    baby2 = dad.copy()
    return baby1, baby2


def CrossOverAlternatingPosition(mum, dad, crossOverRate, chromoLength):
    baby1 = mum.copy()
    baby2 = dad.copy()
    return baby1, baby2


def CrossOverMaximalPositionBased(mum, dad, crossOverRate, chromoLength):
    baby1 = mum.copy()
    baby2 = dad.copy()
    return baby1, baby2


def CrossOverMaximalPositionBased(mum, dad, crossOverRate, chromoLength):
    baby1 = mum.copy()
    baby2 = dad.copy()
    return baby1, baby2


def CrossOverRandomSwapPoint(mum, dad, chromoLength, crossOverRate):
    baby1 = mum.copy()
    baby2 = dad.copy()

    # just return parents as offspring dependent on the rate
    # or if parents are the same
    if random() > crossOverRate:  # or mum['chromo'] == dad['chromo']:
        return baby1, baby2

    crossoverPoint = randint(0, chromoLength)

    baby1['chromo'] = (mum['chromo'][0:crossoverPoint]).copy() + (
        dad['chromo'][crossoverPoint:chromoLength]).copy()
    baby2['chromo'] = (dad['chromo'][0:crossoverPoint]).copy() + (
        mum['chromo'][crossoverPoint:chromoLength]).copy()

    return baby1, baby2


def DecodeRealNumber(chromo, DecodeDict):
    decoded = [DecodeDict[gene] for gene in chromo]
    return decoded


def DecodeString(chromo, DecodeDict):
    return 0


def DecodeBinary(chromo, geneLength, DecodeDict):
    grouped = zip(*[iter(chromo)] * geneLength)
    decoded = [DecodeDict[BinToInt(group)] for group in grouped]

    return decoded


class GA:
    def __init__(self, PopSize=100, CrossOverRate=0.7, MutationRate=0.001, ChromoLength=70, GeneLength=2,
                 selectionType='weightedRouletteWheel', crossOverType='randomSwapPoint', mutateType='Binary',
                 chromoType='Binary', genomeKey='chromo', mutationCustomFunction=[], crossOverCustomFunction=[],
                 setPopFlag=False, stringMutate='swap', DecodeDict=[], numGeneration=100, fitnessTestFunction=[],
                 infoBoard=[], progressGen=[], progressOverall=[]):

        self.ListGenomes = []
        self.PopSize = PopSize
        self.CrossOverRate = CrossOverRate
        self.MutationRate = MutationRate
        self.ChromoLength = ChromoLength
        self.GeneLength = GeneLength

        self.FittestGenome = 0
        self.BestFitnessScore = 0
        self.TotalFitnessScore = 0
        self.Generation = 0

        self.selectionType = selectionType
        self.crossOverType = crossOverType
        self.mutateType = mutateType
        self.chromoType = chromoType
        self.stringMutate = stringMutate
        self.DecodeDict = DecodeDict
        self.genomeKey = genomeKey
        self.MutationCustomFunction = mutationCustomFunction
        self.CrossOverCustomFunction = crossOverCustomFunction
        self.ExternalGenPopulationFlag = setPopFlag
        self.Busy = True

        self.FittestGenome = {'chromo': [],
                              'Fitness': 0,
                              'Info': []
                              }
        self.FittestGenomeEver = self.FittestGenome.copy()
        self.numGeneration = numGeneration
        self.fitnessTestFunction = fitnessTestFunction
        self.infoBoard = infoBoard
        self.progressGen = progressGen
        self.progressOverall = progressOverall
        self.dataMeanStored = []
        self.dataStdStored = []
        self.data = []
        self.x = []
        self.plots = 0

    def CreateStartPopulation(self):

        if not self.ExternalGenPopulationFlag:
            if self.chromoType == 'Binary':
                for genome in range(0, self.PopSize):
                    Genomes = {'chromo': [randint(0, 1) for bit in range(0, self.ChromoLength)],
                               'Fitness': 0,
                               'Info': []
                               }
                    self.ListGenomes.append(Genomes)

                self.FittestGenomeEver = {'chromo': [0 for bit in range(0, self.ChromoLength)],
                                          'Fitness': 0,
                                          'Info': []
                                          }
            if self.chromoType == 'RealNumberInt':
                for genome in range(0, self.PopSize):
                    Genomes = {
                        'chromo': [randint(0, max(self.DecodeDict.keys())) for bit in range(0, self.ChromoLength)],
                        'Fitness': 0,
                        'Info': []
                    }
                    self.ListGenomes.append(Genomes)

                self.FittestGenomeEver = {'chromo': [0 for bit in range(0, self.ChromoLength)],
                                          'Fitness': 0,
                                          'Info': []
                                          }

        self.FittestGenome = self.FittestGenomeEver.copy()
        self.BestFitnessScore = 0
        self.TotalFitnessScore = 0
        self.Generation = 0

    def setStartPopulation(self, population):
        for chromo in population:
            Genomes = {
                'chromo': chromo,
                'Fitness': 0,
                'Info': []
            }
            self.ListGenomes.append(Genomes)

    def MutateChromeRealNumber(self, chromo):
        mutation = chromo.copy
        randomIndex = rd.randrange(len(chromo))
        mutation['chromo'][randomIndex] = PerturbRealNumber(mutation['chromo'][randomIndex], self.MutationRate)
        return mutation

    def MutateChromeBitFlip(self, chromo):
        mutation = chromo.copy()
        randomIndex = rd.randrange(len(chromo))
        mutation['chromo'][randomIndex] = FlipBoolean(mutation['chromo'][randomIndex], self.MutationRate)

        return mutation

    def MutateChromeSwap(self, chromo):
        mutation = chromo.copy()
        randomIndex1 = rd.randrange(len(chromo))
        randomIndex2 = rd.randrange(len(chromo))

        if random() < self.MutationRate:
            mutation['chromo'][randomIndex1], mutation['chromo'][randomIndex2] = \
                mutation['chromo'][randomIndex2], mutation['chromo'][randomIndex1]

        return mutation

    def Mutate(self, chromo):

        if self.mutateType == 'Integer':
            return self.MutateChromeSwap(chromo)

        if self.mutateType == 'RealNumber':
            return self.MutateChromeRealNumber(chromo)

        if self.mutateType == 'Binary':
            return self.MutateChromeBitFlip(chromo)

        if self.mutateType == 'String':
            if self.stringMutate == 'swap':
                return self.MutateChromeSwap(chromo)

    def CrossOverNeuralNetwork(self, mumChrome, dadChrome):
        mumNetwork = mumChrome['chromo']
        dadNetwork = dadChrome['chromo']
        layerRange = min(mumNetwork.NeuralNet['Layers'], dadNetwork.NeuralNet['Layers'])
        numInputs = mumNetwork.numInputs
        numOutputs = mumNetwork.numInputs
        baby1Chromo = NeuralNetwork(numInputs, numOutputs)
        baby2Chromo = NeuralNetwork(numInputs, numOutputs)

        baby1Chromo.NeuralNet['Layers'] = layerRange
        baby2Chromo.NeuralNet['Layers'] = layerRange

        mumForCrossOver = dict(chromo=[])
        dadForCrossOver = dict(chromo=[])
        for layer in range(layerRange+1):
            mumForCrossOver['chromo'] = mumNetwork.FindNeuronsInLayer(layer)
            dadForCrossOver['chromo']= dadNetwork.FindNeuronsInLayer(layer)
            baby1Layer, baby2Layer = self.CrossOver(mumForCrossOver, dadForCrossOver)
            baby1Chromo.NeuralNet['Network'].append(baby1Layer['chromo'][0])
            baby2Chromo.NeuralNet['Network'].append(baby2Layer['chromo'][0])

        baby1 = {'chromo': baby1Chromo,
                 'Fitness': 0,
                 'Info': []
                 }

        baby2 = {'chromo': baby2Chromo,
                 'Fitness': 0,
                 'Info': []
                 }

        return baby1, baby2

    def CrossOver(self, mumForCrossOver, dadForCrossOver):

        if self.crossOverType == 'randomSwapPoint':
            return CrossOverRandomSwapPoint(mumForCrossOver, dadForCrossOver, self.ChromoLength,
                                            crossOverRate=self.CrossOverRate)

        if self.crossOverType == 'partiallyMapped':
            return CrossOverPartiallyMapped(mumForCrossOver, dadForCrossOver, self.ChromoLength,
                                            crossOverRate=self.CrossOverRate)

    def Selection(self):
        if self.selectionType == 'rouletteWheel':
            return self.RouletteWheelSelection()

        if self.selectionType == 'weightedRouletteWheel':
            return self.WeightedRouletteWheelSelection()

    def RouletteWheelSelection(self):
        selectedGenome = []
        Slice = random() * self.TotalFitnessScore

        total = 0.0

        for genome in self.ListGenomes:
            total += genome['Fitness']

            if total > Slice:
                selectedGenome = genome.copy()
                break

        return selectedGenome

    def WeightedRouletteWheelSelection(self):
        # Find genome slice
        RouletteWheel = []

        for index, genome in enumerate(self.ListGenomes):
            # if self.genomeKey == 'Network':
            #    Slice = math.ceil((genome.NeuralNet['Fitness'] * 100) / self.TotalFitnessScore)
            # else:
            #     Slice = math.ceil((genome['Fitness'] * 100) / self.TotalFitnessScore)

            Slice = math.ceil((genome['Fitness'] * 100) / self.TotalFitnessScore)
            for i in range(0, Slice):
                RouletteWheel.append(index)

        selected = rd.choice(RouletteWheel)
        selectedGenome = self.ListGenomes[selected]
        return selectedGenome

    def UpdateFitnessScores(self, fitnessTestFunction, genomeKey):
        self.BestFitnessScore = 0
        self.TotalFitnessScore = 0
        # print(f'Update :{self.ListGenomes}')
        for genome in self.ListGenomes:
            # if genomeKey == 'Network':
            #   decode = self.Decode(genome)
            # else:
            #    decode = self.Decode(genome[genomeKey])
            decode = self.Decode(genome)
            fitness, info = fitnessTestFunction(decode)
            # genome.NeuralNet['Fitness'], genome.NeuralNet['Info'] = fitnessTestFunction(decode)
            # print( f"Update:   {genome['Fitness']} {genome['Info']}")
            self.TotalFitnessScore += fitness
            self.data.append(fitness)
            genome['Fitness'] = fitness
            genome['Info'] = info

            if fitness > self.BestFitnessScore:
                self.BestFitnessScore = fitness
                self.FittestGenome = copy.deepcopy(genome)

                if fitness == 1:
                    self.Busy = False

            # if genomeKey == 'Network':
            # genome.NeuralNet['Fitness'] = fitness
            # genome.NeuralNet['Info'] = info

            if self.FittestGenome['Fitness'] > self.FittestGenomeEver['Fitness']:
                self.FittestGenomeEver = copy.deepcopy(self.FittestGenome)
            # else:
            #   genome['Fitness'] = fitness
            #   genome['Info'] = info

            if self.FittestGenome['Fitness'] > self.FittestGenomeEver['Fitness']:
                self.FittestGenomeEver = copy.deepcopy(self.FittestGenome)
        # print(f"----{self.FittestGenomeEver['Fitness']} {self.FittestGenomeEver['Info']}")

    @staticmethod
    def CalculateFitnessScore(fitnessTestFunction, decode):
        return fitnessTestFunction(decode)

    def GenerateBabies(self):
        NewBabies = 0
        ListBabyGenomes = []

        while NewBabies < self.PopSize:
            mumSelected = self.Selection()
            dadSelected = self.Selection()

            if not self.CrossOverCustomFunction:
                baby1, baby2 = self.CrossOver(mumSelected, dadSelected)
            else:
                baby1, baby2 = self.CrossOverNeuralNetwork(mumSelected, dadSelected)

            if not self.MutationCustomFunction:
                baby1 = self.Mutate(baby1)
                baby2 = self.Mutate(baby2)
            else:
                baby1Chromo = self.MutationCustomFunction(baby1['chromo'])
                baby2Chromo = self.MutationCustomFunction(baby2['chromo'])

                baby1 = {'chromo': baby1Chromo,
                         'Fitness': 0,
                         'Info': []
                         }

                baby2 = {'chromo': baby2Chromo,
                         'Fitness': 0,
                         'Info': []
                         }

            ListBabyGenomes.append(baby1)
            ListBabyGenomes.append(baby2)
            NewBabies += 2

        self.ListGenomes = ListBabyGenomes.copy()
        self.ListGenomes.append(self.FittestGenomeEver.copy())

    def Decode(self, chromo):

        if self.chromoType == 'RealNumberInt':
            return DecodeRealNumber(chromo, self.DecodeDict)

        if self.chromoType == 'Binary':
            return DecodeBinary(chromo, self.GeneLength, self.DecodeDict)

        if self.chromoType == 'String':
            return DecodeString(chromo, self.DecodeDict)

        if self.chromoType == 'Network':
            return chromo['chromo']

    def Epoch(self):

        while self.Generation != self.numGeneration:
            self.UpdateFitnessScores(self.fitnessTestFunction, self.genomeKey)
            self.GenerateBabies()
            self.Generation += 1

            self.InfoBoard()
            self.PlotData()
            print('------')

        print('done with loop')

    def Evolve(self):
        self.CreateStartPopulation()
        self.Epoch()

    def Log(self):
        pass

    def Statistics(self):
        dataMean = np.mean(self.data)
        dataStd = np.std(self.data)
        bins = np.array(range(0, 100)) / 100  # np.array(range(0, len(self.data))) / len(self.data)
        self.x.append(self.Generation - 1)

        self.dataMeanStored.append(dataMean)
        self.dataStdStored.append(dataStd)

        return dataMean, dataStd, self.x, bins

    def PlotData(self):
        dat = []
        if self.plots == 3:
            self.plots = 0
            plt.clf()

        self.plots += 1
        fig = plt.figure(1)
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)

        ax1.set_xlim([0, 2])
        meanPlotData, = ax2.plot(dat)
        ax2.set_xlim([0, self.numGeneration])
        stdPlotData, = ax3.plot(dat)
        ax3.set_xlim([0, self.numGeneration])
        GenData, = ax4.plot(dat)
        ax4.set_xlim([0, len(self.data)])
        plt.ion()
        plt.show()

        dataMean, dataStd, x, bins = self.Statistics()
        ax1.hist(self.data, bins, histtype='bar', rwidth=10)
        meanPlotData.set_ydata(self.dataMeanStored)
        meanPlotData.set_xdata(x)
        stdPlotData.set_ydata(self.dataStdStored)
        stdPlotData.set_xdata(x)
        GenData.set_ydata(self.data)
        GenData.set_xdata(range(0, len(self.data)))
        plt.pause(0.00001)
        self.data = []

    def InfoBoard(self):
        # if self.chromoType == 'Network':
        #     info = self.FittestGenome['Info']
        #     info_Ever = self.FittestGenomeEver['Info']
        #     fitness = self.FittestGenome['Fitness']
        # else:
        #     info = self.FittestGenome['Info']
        #     info_Ever = self.FittestGenomeEver['Info']
        #     fitness = self.FittestGenome['Fitness']

        info = self.FittestGenome['Info']
        info_Ever = self.FittestGenomeEver['Info']
        fitness = self.FittestGenome['Fitness']

        self.progressGen(info, fittestEver=False)
        self.progressOverall(info_Ever, fittestEver=True)
        self.infoBoard(f"Gen= {self.Generation} "
                       f"best score ={self.BestFitnessScore:.3f}={fitness:.3f}"
                       f" Total fitness={self.TotalFitnessScore:.3f}"
                       f" fitness Info {info}"
                       f" fitness Ever Info {info_Ever}"
                       )


if __name__ == '__main__':
    print("GA")
    mum = [2, 5, 0, 3, 6, 1, 4, 7]
    dad = [3, 4, 0, 7, 2, 5, 1, 6]
    print(mum)
    print(CrossOverPartiallyMapped(mum, dad, len(mum), crossOverRate=0.7))
