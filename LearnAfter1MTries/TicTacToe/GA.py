import random as rd
from random import random
from random import randint
import math


class GA:
    def __init__(self, PopSize=1000, CrossOverRate=0.7, MutationRate=0.001, ChromoLength=70, GeneLength=2):
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
        self.Busy = True

    def CreateStartPopulation(self):
        for genome in range(0, self.PopSize):
            Genomes = {'chromo': [randint(0, 1) for bit in range(0, self.ChromoLength)],
                       'Fitness': 0,

                       }
            self.ListGenomes.append(Genomes)

        self.FittestGenomeEver = {'chromo': [0 for bit in range(0, self.ChromoLength)],
                                  'Fitness': 0,
                                  }
        self.FittestGenome = self.FittestGenomeEver.copy()
        self.BestFitnessScore = 0
        self.TotalFitnessScore = 0
        self.Generation = 0

    def Mutate(self, chromo):
        mutation = chromo
        for index, bit in enumerate(chromo):
            if random() < self.MutationRate:
                if bit == 1:
                    mutation[index] = 0
                else:
                    mutation[index] = 1

        return mutation

    def CrossOver(self, mum, dad):
        baby1 = mum.copy()
        baby2 = dad.copy()

        # just return parents as offspring dependent on the rate
        # or if parents are the same
        if random() > self.CrossOverRate or mum['chromo'] == dad['chromo']:
            baby1 = mum.copy()
            baby2 = dad.copy()
            return baby1, baby2

        # determine a crossover point
        crossoverPoint = randint(0, self.ChromoLength)

        baby1['chromo'] = (mum['chromo'][0:crossoverPoint]).copy() + (
            dad['chromo'][crossoverPoint:self.ChromoLength]).copy()
        baby2['chromo'] = (dad['chromo'][0:crossoverPoint]).copy() + (
            mum['chromo'][crossoverPoint:self.ChromoLength]).copy()

        baby1['Fitness'] = 0
        baby2['Fitness'] = 0

        return baby1, baby2

    def RouletteWheelSelection(self):
        slice = random() * self.TotalFitnessScore
        # print("slice--{}--{}".format(slice,self.TotalFitnessScore))
        total = 0.0
        # selectedGenome=self.ListGenomes[0].copy()

        for genome in self.ListGenomes:
            total += genome['Fitness']

            if total > slice:
                selectedGenome = genome.copy()
                break

        return selectedGenome

    def WeightedRouletteWheelSelection(self):
        # Find genome slice
        RouletteWheel = []
        for genome in self.ListGenomes:
            slice = math.ceil((genome['Fitness'] * 100) / self.TotalFitnessScore)
            # print("total Fitness {} Fitness {} slice ={}".format(self.TotalFitnessScore,genome['Fitness'],slice))

            for index in range(0, slice):
                RouletteWheel.append(index)
        # print("------RouletteWheel{}".format(len(RouletteWheel)))
        selected = rd.choice(RouletteWheel)
        selectedGenome = self.ListGenomes(selected)
        return selectedGenome

    def UpdateFitnessScores(self, fitnessTest):
        self.BestFitnessScore = 0
        self.TotalFitnessScore = 0

        for genome in self.ListGenomes:
            # decode each genomes chromosome into a vector of directions
            decode = self.Decode(genome['chromo'])

            # get it's fitness score
            genome['Fitness'] = fitnessTest.test(decode)
            # print("genome {}".format(genome))
            self.TotalFitnessScore += genome['Fitness']

            if genome['Fitness'] > self.BestFitnessScore:
                self.BestFitnessScore = genome['Fitness']
                self.FittestGenome = genome.copy()
                # print("BestFitnessScore {} genome {}  FittestEver {}".format(self.BestFitnessScore,genome['Fitness'],self.FittestGenomeEver['Fitness']))

                if genome['Fitness'] == 1:
                    self.Busy = False

            if self.FittestGenome['Fitness'] > self.FittestGenomeEver['Fitness']:
                self.FittestGenomeEver = self.FittestGenome.copy()

    def Decode(self, genome):
        i = -1
        gene = []
        direction = []

        for chromo in genome:
            i += 1
            gene.append(chromo)

            if i == 1:
                direction.append(self.BinToInt(gene))
                i = -1
                gene = []
        return direction

    def BinToInt(self, bin):
        val = 0
        multiplier = 1
        # print("{} bin".format(bin))
        for bit in reversed(bin):
            val += bit * multiplier
            multiplier *= 2

        return val

    def Epoch(self, fitnessTest):
        # print("BestBU{}".format(test.FittestGenome))
        self.UpdateFitnessScores(fitnessTest)
        # print("BestAU{}".format(test.FittestGenome))
        NewBabies = 0
        ListBabyGenomes = []

        while NewBabies < self.PopSize:
            mum = self.WeightedRouletteWheelSelection()
            dad = self.WeightedRouletteWheelSelection()

            baby1, baby2 = self.CrossOver(mum, dad)

            baby1['chromo'] = self.Mutate(baby1['chromo'].copy())
            baby2['chromo'] = self.Mutate(baby2['chromo'].copy())

            ListBabyGenomes.append(baby1)
            ListBabyGenomes.append(baby2)
            NewBabies += 2

        self.ListGenomes = ListBabyGenomes.copy()
        self.ListGenomes.append(self.FittestGenome.copy())
        # print("Adding{} ---{}".format(self.ListGenomes[-1],self.FittestGenome.copy()))
        self.Generation += 1
