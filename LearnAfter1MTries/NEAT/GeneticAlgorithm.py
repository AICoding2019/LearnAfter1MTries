import random as rd
from random import random
from random import randint
import math


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
    mutation = chromo.copy
    randomIndex = rd.randrange(len(chromo))
    if random() < mutationRate:
        if mutation[randomIndex] == 1:
            mutation[randomIndex] = 0
        else:
            mutation[randomIndex] = 1

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
    return baby1, baby2


def CrossOverOrder(chromo):
    return 0


def CrossOverAlternatingPosition(chromo):
    return 0


def CrossOverMaximalPreservation(chromo):
    return 0


def CrossOverMaximalPositionBased(chromo):
    return 0


def CrossOverMaximalPositionBased(chromo):
    return 0


def CrossOverRandomSwapPoint(mum, dad, crossOverRate, chromoLength):
    baby1 = mum.copy()
    baby2 = dad.copy()

    # just return parents as offspring dependent on the rate
    # or if parents are the same
    if random() > crossOverRate or mum['chromo'] == dad['chromo']:
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


def CrossOverRealNumber(chromo):
    return 0


def DecodeRealNumber(chromo, DecodePrototype):
    return 0


def DecodeString(chromo, DecodePrototype):
    return 0


def DecodeBinary(chromo, DecodePrototype):
    return 0


class GA:
    def __init__(self, PopSize=1000, CrossOverRate=0.7, MutationRate=0.001, ChromoLength=70, GeneLength=2,
                 selectionType='WeightedRouletteWheel', crossOverType='randomSwapPoint', mutateType='Binary',
                 stringMutate='swap', DecodePrototype=[]):

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
        self.stringMutate = stringMutate
        self.DecodePrototype = DecodePrototype
        self.Busy = True

        self.FittestGenomeEver = []

    def CreateStartPopulation(self):
        pass

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

        if self.selectionType == 'WeightedRouletteWheel':
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
        for genome in self.ListGenomes:
            Slice = math.ceil((genome['Fitness'] * 100) / self.TotalFitnessScore)

            for index in range(0, Slice):
                RouletteWheel.append(index)

        selected = rd.choice(RouletteWheel)
        selectedGenome = self.ListGenomes[selected]
        return selectedGenome

    def UpdateFitnessScores(self, fitnessTest):
        self.BestFitnessScore = 0
        self.TotalFitnessScore = 0

        for genome in self.ListGenomes:
            decode = self.Decode(genome['chromo'])

            genome['Fitness'] = fitnessTest.test(decode)
            self.TotalFitnessScore += genome['Fitness']

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
            mum, dad = self.Selection()
            baby1, baby2 = self.CrossOver(mum, dad)
            baby1 = self.Mutate(baby1)
            baby2 = self.Mutate(baby2)

            ListBabyGenomes.append(baby1)
            ListBabyGenomes.append(baby2)
            NewBabies += 2

            self.ListGenomes = ListBabyGenomes.copy()
            self.ListGenomes.append(self.FittestGenomeEver.copy())

    def Decode(self, chromo, chromoType='RealNumber'):

        if chromoType == 'RealNumber':
            return DecodeRealNumber(chromo, self.DecodePrototype)

        if chromoType == 'Binary':
            return DecodeBinary(chromo, self.DecodePrototype)

        if chromoType == 'String':
            return DecodeString(chromo, self.DecodePrototype)

    def Epoch(self, fitnessTest):
        self.UpdateFitnessScores(fitnessTest)
        self.GenerateBabies()
        self.Generation += 1

    def Log(self):
        pass


if __name__ == '__main__':
    print("GA")
