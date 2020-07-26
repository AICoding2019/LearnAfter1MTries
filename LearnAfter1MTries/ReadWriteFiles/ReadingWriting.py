import numpy as np
import h5py
from os import mkdir, path, rename ,remove
from shutil import rmtree

def DeleteFile(fileName):
    remove(fileName)
    print("File Removed!")


def createChildrenFile(child, layer):
    with h5py.File('Children.hdf5', 'a') as children:
        children.create_group(child + '/' + layer + '/Weight')
        children.create_group(child + '/' + layer + '/Bias')
        children.create_group(child + '/' + layer + '/RecurrentWeight')
        children.create_group(child + '/' + layer + '/Enable')
        children.create_group(child + '/' + layer + '/RecurrentEnable')
        children.create_group(child + '/' + layer + '/Activation')
        children.create_group(child + '/' + layer + '/Output')
        children.create_group(child + '/' + layer + '/Input')


def AddChildData(path, weight, bias, recurrentWeight, enable, recurrentEnable, activation, In, Out):
    with h5py.File('Children.hdf5','a') as children:
        children.create_dataset(path + '/Weight/default', data=weight)
        children.create_dataset(path + '/Bias/default', data=bias)
        children.create_dataset(path + '/RecurrentWeight/default', data=recurrentWeight)
        children.create_dataset(path + '/Enable/default', data=enable)
        children.create_dataset(path + '/RecurrentEnable/default', data=recurrentEnable)
        children.create_dataset(path + '/Activation/default', data=activation)
        children.create_dataset(path + '/Output/default', data=Out)
        children.create_dataset(path + '/Input/default', data=In)

DeleteFile('Children.hdf5')

createChildrenFile('Child0', 'Layer0')
createChildrenFile('Child0', 'Layer1')
createChildrenFile('Child1', 'Layer0')
createChildrenFile('Child1', 'Layer1')
child = 'Child0'
layer = 'Layer0'
path = child+'/'+layer
weight = np.random.rand(1000)
bias = np.random.rand(1)
recurrentWeight = np.random.rand(1)
enable = np.array(1)
recurrentEnable = np.array(1)
activation = np.array(1)
In = np.random.rand(1000)
Out = np.random.rand(1000)
AddChildData(path, weight, bias, recurrentWeight, enable, recurrentEnable, activation, In, Out)
#
child = 'Child0'
layer = 'Layer1'
path = child+'/'+layer
weight = np.random.rand(1000)
bias = np.random.rand(1)
recurrentWeight = np.random.rand(1)
enable = np.array(1)
recurrentEnable = np.array(1)
activation = np.array(1)
In = np.random.rand(1000)
Out = np.random.rand(1000)
AddChildData(path, weight, bias, recurrentWeight, enable, recurrentEnable, activation, In, Out)
#
child = 'Child1'
layer = 'Layer0'
path = child+'/'+layer
weight = np.random.rand(1000)
bias = np.random.rand(1)
recurrentWeight = np.random.rand(1)
enable = np.array(1)
recurrentEnable = np.array(1)
activation = np.array(1)
In = np.random.rand(1000)
Out = np.random.rand(1000)
AddChildData(path, weight, bias, recurrentWeight, enable, recurrentEnable, activation, In, Out)
#
child = 'Child1'
layer = 'Layer1'
path = child+'/'+layer
weight = np.random.rand(1000)
bias = np.random.rand(1)
recurrentWeight = np.random.rand(1)
enable = np.array(1)
recurrentEnable = np.array(1)
activation = np.array(1)
In = np.random.rand(1000)
Out = np.random.rand(1000)
path = child+'/'+layer
AddChildData(path, weight, bias, recurrentWeight, enable, recurrentEnable, activation, In, Out)


with h5py.File('Children.hdf5', 'r') as f:
    for key in f['Child0']['Layer0'].keys():
        print(key)
