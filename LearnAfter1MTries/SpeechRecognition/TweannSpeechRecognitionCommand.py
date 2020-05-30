from LearnAfter1MTries.TWEANN.TweannRunner import *
from random import shuffle
import json
from scipy.io import wavfile
import numpy as np

nsamples = 16000
path = '/home/a/Downloads/speech_commands'
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
    return sum(np.array(y) - np.array(x)) / len(y)


def TestFitness(NN):

    numberCorrect = 0
    shuffle(data['training'])
    MAE = 1E6
    for wavFileName in data['training']:
        X = get_wav(path+'/DataSet/' + wavFileName)
        NN.UpdateGraph(X)

        label = wavFileName.split('/')[1]
        y = data['OneHotEncode'][label]
        MAE = meanAbsoluteError(y, NN.NeuralNet['Output'])

        prediction = np.argmax(NN.NeuralNet['Output'])
        actual = np.argmax(y)

        if prediction == actual:
            numberCorrect += 1

        NN.NeuralNet['Output'] = []

    fitness = (numberCorrect / len(data['training']) + (1 / (1 + MAE))) / 2
    NN.NeuralNet['Fitness'] = fitness

    return fitness, [numberCorrect, NN.NeuralNet['Species']]


testSpeechRecog = Tweann(numInputs=nsamples, numOutputs=classLength, PopSize=10, numGeneration=100000,
                         selectionType='rouletteWheel', fitnessTestFunction=TestFitness)
testSpeechRecog.Evolve()

while True:
    pass
