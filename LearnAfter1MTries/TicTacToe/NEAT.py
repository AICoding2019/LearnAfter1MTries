
class Genome:
    def __init__(self, PopSize=1000, CrossOverRate=0.7, MutationRate=0.001, ChromosomeLength=70, GeneLength=2):
        self.LinkGene = {'FromNeuron': [],
                         'ToNeuron': [],
                         'Weight': [],
                         'Enabled': [],
                         'Recurrent': [],
                         'InnovationID': [],
                         }

        self.NeuronGene = {'ID': [],
                           'NeuronType': [],
                           'Recurrent': [],
                           'ActivationResponse': [],
                           'SplitY': [],
                           'SplitX': [],
                           }

        # Private variables
        self.__GenomeID = []
        self.__ListNeurons = []
        self.__ListLinks = []
        self.__Phenotype = []
        self.__Fitness = []
        self.__AdjustedFitness = []
        self.__AmountToSpawn = []
        self.__NumInputs = []
        self.__NumOutputs = []
        self.__Species = []

    def __DuplicateLink(self, NeuronIn, NeuronOut):
        pass

    def __GetElementPos(self, neuronID):
        pass

    def __AlreadyHaveThisNeuronID(self, ID):
        pass

    def CreatePhenotype(self, depth):
        pass

    def DeletePhenotype(self):
        pass

    def AddLink(self, MutationRate, ChanceOfRecurrent, Innovation, NumTryToFindLoop, NumTryToFindOldLink):
        pass

    def MutateWeights(self, MutateRate, ProbNewMutate, MaxPerturbation):
        pass

    def MutateActivationResponse(self, MutateRate, MaxPerturbation):
        pass

    def GetCompatibilityScore(self, genome):
        pass

    def SortGenes(self):
        pass


class GA:
    def __init__(self):
        pass
