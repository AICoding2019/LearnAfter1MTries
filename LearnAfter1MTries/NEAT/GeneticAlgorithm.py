import random as rd
from random import random
from random import randint
import math
import matplotlib.pyplot as plt
import numpy as np


def BinToInt(binary):
    val = 0
    multiplier = 1

    for bit in reversed(binary):
        val += bit * multiplier
        multiplier *= 2

    return val


def MutateRealNumber(chromo, mutationRate):
    mutation = chromo.copy
    randomIndex = rd.randrange(len(chromo))
    if random() < mutationRate:
        mutation[randomIndex] += random.choice[-1, 1] * mutation[randomIndex] * 0.001  # mutation is +/- 0.1%

    return mutation


def MutateBitFlip(chromo, mutationRate):
    mutation = chromo.copy()
    randomIndex = rd.randrange(len(chromo))
    if random() < mutationRate:
        if mutation['chromo'][randomIndex] == 1:
            mutation['chromo'][randomIndex] = 0
        else:
            mutation['chromo'][randomIndex] = 1

    return mutation


def MutateRandomResetting(chromo, mutationRate, chromoRange):
    """"""
    '''https: // www.geeksforgeeks.org / mutation - algorithms -
    ''for -string - manipulation - ga /
    '''''''''
    mutation = chromo.copy
    randomIndex = rd.randrange(len(chromo))

    if random() < mutationRate:
        mutation[randomIndex] = random.choice(chromoRange)

    return mutation


def MutateSwap(chromo, mutationRate):
    mutation = chromo.copy
    randomIndex1 = rd.randrange(len(chromo))
    randomIndex2 = rd.randrange(len(chromo))

    if random() < mutationRate:
        mutation[randomIndex1], mutation[randomIndex2] = mutation[randomIndex2], mutation[randomIndex1]

    return mutation


def CrossOverPartiallyMapped(mum, dad, crossOverRate, chromoLength):
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


def CrossOverMaximalPreservation(mum, dad, crossOverRate, chromoLength):
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


def CrossOverRandomSwapPoint(mum, dad, crossOverRate, chromoLength):
    baby1 = mum.copy()
    baby2 = dad.copy()

    # just return parents as offspring dependent on the rate
    # or if parents are the same
    if random() > crossOverRate: #or mum['chromo'] == dad['chromo']:
        baby1 = mum.copy()
        baby2 = dad.copy()
        return baby1, baby2

    crossoverPoint = randint(0, chromoLength)

    baby1['chromo'] = (mum['chromo'][0:crossoverPoint]).copy() + (
        dad['chromo'][crossoverPoint:chromoLength]).copy()
    baby2['chromo'] = (dad['chromo'][0:crossoverPoint]).copy() + (
        mum['chromo'][crossoverPoint:chromoLength]).copy()

    baby1['Fitness'] = 0
    baby2['Fitness'] = 0

    return baby1, baby2


def CrossOverRealNumber(mum, dad, crossOverRate, chromoLength):
    baby1 = mum.copy()
    baby2 = dad.copy()
    return baby1, baby2


def DecodeRealNumber(chromo, DecodeDict):
    return 0


def DecodeString(chromo, DecodeDict):
    return 0


def DecodeBinary(chromo, geneLength, DecodeDict):
    grouped = zip(*[iter(chromo)] * geneLength)
    decoded = [DecodeDict[BinToInt(group)] for group in grouped]

    return decoded


class GA:
    def __init__(self, PopSize=100, CrossOverRate=0.7, MutationRate=0.001, ChromoLength=70, GeneLength=2,
                 selectionType='weightedRouletteWheel', crossOverType='randomSwapPoint', mutateType='Binary',
                 chromoType='Binary',
                 stringMutate='swap', DecodeDict=[], numGeneration=100, fitnessTest=[],
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
        self.Busy = True

        self.FittestGenome = []
        self.FittestGenomeEver = []
        self.numGeneration = numGeneration
        self.fitnessTest = fitnessTest
        self.infoBoard = infoBoard
        self.progressGen = progressGen
        self.progressOverall = progressOverall
        self.dataMeanStored = []
        self.dataStdStored = []
        self.data = []
        self.x = []
        self.plots =0

    def CreateStartPopulation(self):
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
        self.FittestGenome = self.FittestGenomeEver.copy()
        self.BestFitnessScore = 0
        self.TotalFitnessScore = 0
        self.Generation = 0

    def Mutate(self, chromo):

        if self.mutateType == 'RealNumber':
            return MutateRealNumber(chromo, self.MutationRate)

        if self.mutateType == 'Binary':
            return MutateBitFlip(chromo, self.MutationRate)

        if self.mutateType == 'String':
            if self.stringMutate == 'swap':
                return MutateSwap(chromo, self.MutationRate)

    def CrossOver(self, mum, dad):
        if self.crossOverType == 'randomSwapPoint':
            return CrossOverRandomSwapPoint(mum, dad, self.CrossOverRate, self.ChromoLength)

        if self.crossOverType == 'partiallyMapped':
            return CrossOverPartiallyMapped(mum, dad, self.CrossOverRate, self.ChromoLength)

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
            Slice = math.ceil((genome['Fitness'] * 100) / self.TotalFitnessScore)

            for i in range(0, Slice):
                RouletteWheel.append(index)

        selected = rd.choice(RouletteWheel)
        selectedGenome = self.ListGenomes[selected]
        return selectedGenome

    def UpdateFitnessScores(self, fitnessTest):
        self.BestFitnessScore = 0
        self.TotalFitnessScore = 0

        for genome in self.ListGenomes:
            decode = self.Decode(genome['chromo'])

            genome['Fitness'], genome['Info'] = fitnessTest(decode)
            self.TotalFitnessScore += genome['Fitness']
            self.data.append(genome['Fitness'])

            if genome['Fitness'] > self.BestFitnessScore:
                self.BestFitnessScore = genome['Fitness']
                self.FittestGenome = genome.copy()

                if genome['Fitness'] == 1:
                    self.Busy = False

            if self.FittestGenome['Fitness'] > self.FittestGenomeEver['Fitness']:
                self.FittestGenomeEver = self.FittestGenome.copy()

    def GenerateBabies(self):
        NewBabies = 0
        ListBabyGenomes = []

        while NewBabies < self.PopSize:
            mum = self.Selection()
            dad = self.Selection()
            baby1, baby2 = self.CrossOver(mum, dad)
            baby1 = self.Mutate(baby1)
            baby2 = self.Mutate(baby2)

            ListBabyGenomes.append(baby1)
            ListBabyGenomes.append(baby2)
            NewBabies += 2

        self.ListGenomes = ListBabyGenomes.copy()
        self.ListGenomes.append(self.FittestGenomeEver.copy())

    def Decode(self, chromo):

        if self.chromoType == 'RealNumber':
            return DecodeRealNumber(chromo, self.DecodeDict)

        if self.chromoType == 'Binary':
            return DecodeBinary(chromo, self.GeneLength, self.DecodeDict)

        if self.chromoType == 'String':
            return DecodeString(chromo, self.DecodeDict)

    def Epoch(self, fitnessTest):

        while self.Generation != self.numGeneration:
            self.UpdateFitnessScores(fitnessTest)
            self.GenerateBabies()
            self.Generation += 1

            self.InfoBoard()
            self.PlotData()

        print('done with loop')

    def Evolve(self):
        self.CreateStartPopulation()
        self.Epoch(self.fitnessTest)

    def Log(self):
        pass

    def Statistics(self):
        dataMean = np.mean(self.data)
        dataStd = np.std(self.data)
        bins = np.array(range(0, len(self.data))) / len(self.data)
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

        ax1.set_xlim([0, 1.1])
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
        self.progressGen(self.FittestGenome['Info'], fittestEver=False)
        self.progressOverall(self.FittestGenomeEver['Info'], fittestEver=True)
        self.infoBoard(f"Generation= {self.Generation} "
                       f"best fitness score ={self.BestFitnessScore:.3f}={self.FittestGenome['Fitness']:.3f}"
                       f" Total fitness={self.TotalFitnessScore:.3f}"
                       f" fitness Info{self.FittestGenome['Info']}"
                       f" fitness Ever Info{self.FittestGenomeEver['Info']}"
                       )


if __name__ == '__main__':
    print("GA")
