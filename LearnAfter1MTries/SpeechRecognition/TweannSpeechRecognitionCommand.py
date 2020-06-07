from LearnAfter1MTries.TWEANN.TweannRunner import *
from random import shuffle
import json
from scipy.io import wavfile
import numpy as np

nsamples = 16000
path = '/home/azureuser/Downloads/data/speech_commands/'
dataSetName = 'SpeechDataSet.json'

with open(path + '/' + dataSetName, 'r') as json_file:
    data = json.load(json_file)

classLength = len(data['classes'])


def get_wav(file_name, nsamples=16000):
    wav = wavfile.read(file_name)[1]
    if wav.size < nsamples:
        d = np.pad(wav, (nsamples - wav.size, 0), mode='constant')
    else:
        d = wav[0:nsamples]
    return d


def meanAbsoluteError(y, x):
    return sum(np.fabs(np.array(y) - np.array(x))) / len(y)


def TestFitness(NN):

    numberCorrect = 0
    shuffle(data['training'])

    totalError =0
    for wavFileName in data['training'][0:10]:
        X = get_wav(path+'/DataSet/' + wavFileName)
        NN.UpdateGraph(X)

        label = wavFileName.split('/')[1]
        y = data['OneHotEncode'][label]
        MAE = meanAbsoluteError(y, NN.NeuralNet['Output'])
        totalError += MAE

        prediction = np.argmax(NN.NeuralNet['Output'])
        actual = np.argmax(y)

        if prediction == actual:
            numberCorrect += 1

        NN.NeuralNet['Output'] = []

    fitness = (10*numberCorrect + (1 / (1 + totalError))) / 11
    NN.NeuralNet['Fitness'] = fitness

    return fitness, [numberCorrect, totalError, NN.NeuralNet['Species']]


testSpeechRecog = Tweann(numInputs=nsamples, numOutputs=classLength, PopSize=100, numGeneration=100000,
                         selectionType='rouletteWheel', fitnessTestFunction=TestFitness,displayer=False)
testSpeechRecog.Evolve()

while True:
    pass
