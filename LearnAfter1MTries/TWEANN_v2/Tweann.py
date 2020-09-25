import numpy as np
import h5py
from os import remove
from random import choice
from math import exp, tanh


class Tweann:
    def __init__(self, num_inputs, num_outputs, population_size=100, crossover_rate=0.7, mutation_rate=0.1,
                 add_node_mutation_rate=0.1, selection_type='roulette_wheel',
                 num_generation=2, log_path=None):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_type = selection_type
        self.num_generation = num_generation
        self.add_node_mutation_rate = add_node_mutation_rate
        self.log_path = log_path

        self.fittest_child = None
        self.fittest_child_ever = None
        self.best_fitness_score = 0
        self.best_fitness_score_ever = 0
        self.activationFunc = ['sigmoid', 'relu', 'tanh', 'leakyRelu', 'linear']
        self.neuron_chromo = ['Weights', 'Bias', 'Enabled', 'Recurrent', 'RecurrentWeight', 'Activation', 'Inputs',
                              'Output']

    def create_children_file(self):
        pass

    def evolve(self):
        print("Creating population")
        self.create_population()
        print("Population created")
        self.epoch()

    def create_population(self):
        for child in range(0, self.population_size):
            child_init_data = self.create_init_child_chromo('Child' + str(child))
            self.create_child_dataset(child_init_data)

    def create_init_child_chromo(self, child):
        chromo = dict()
        for output_node_num in range(0, self.num_outputs):
            node_name = 'Node' + str(output_node_num)
            chromo[child + '/Layer-1/' + node_name + '/Weights'] = np.random.rand(self.num_inputs, 1)
            chromo[child + '/Layer-1/' + node_name + '/Bias'] = np.random.rand(1, 1)
            chromo[child + '/Layer-1/' + node_name + '/Enabled'] = 1
            chromo[child + '/Layer-1/' + node_name + '/Recurrent'] = choice([0, 1])
            chromo[child + '/Layer-1/' + node_name + '/RecurrentWeight'] = np.random.rand(1, 1)
            chromo[child + '/Layer-1/' + node_name + '/Activation'] = choice(self.activationFunc)
            chromo[child + '/Layer-1/' + node_name + '/Inputs'] = np.zeros((self.num_inputs, 1))
            chromo[child + '/Layer-1/' + node_name + '/Output'] = np.zeros((1, 1))

        chromo[child + '/Layer-1/Nodes'] = self.num_outputs
        chromo[child + '/HiddenLayers'] = 0
        chromo[child + '/Fitness'] = 0
        return chromo

    @staticmethod
    def create_child_dataset(child_dataset):
        with h5py.File('Children.hdf5', 'a') as children:
            for key in child_dataset:
                group = children.create_group(key)
                group.create_dataset('data', data=child_dataset[key])

    # Todo
    @staticmethod
    def write_to_child_dataset(child_dataset):
        with h5py.File('Children.hdf5', 'r+') as children:
            for key in child_dataset:
                children.create_dataset('data', data=child_dataset[key])

    def epoch(self):
        generation = 0
        while generation < self.num_generation:
            generation += 1
            self.update_fitness(self.fitness_function)

    def update_fitness(self):
        pass

    def get_neural_network_output(self, child, nn_input):
        self._update_neural_network(child, nn_input)
        output_layer = child + '/' + 'Layer-1'
        return self._get_layer_output(output_layer)

    def _update_neural_network(self, child, nn_input):
        num_hidden_layers = self._read_data(child + '/HiddenLayers')
        self._update_input_layer(child, num_hidden_layers, nn_input)
        self._update_hidden_layers(child, num_hidden_layers)
        self._update_output_layer(child)

    def _update_input_layer(self, child, num_hidden_layers, nn_input):
        if num_hidden_layers == 0:
            start_layer = child + '/Layer-1'
        else:
            start_layer = child + '/Layer0'
        self._set_layer_input(start_layer, nn_input)

    def _update_hidden_layers(self, child, num_hidden_layers):
        for num_layer in range(0, num_hidden_layers):
            current_layer = child + '/Layer' + str(num_layer)
            num_nodes = self._read_data(current_layer + '/Nodes')
            self._update_current_layer_outputs(current_layer, num_nodes)

            if num_layer < num_hidden_layers - 1:
                next_layer = child + '/Layer' + str(num_layer + 1)
                self._update_next_layer_inputs(current_layer, next_layer)

    def _update_current_layer_outputs(self, layer, num_nodes):
        for num_node in range(0, num_nodes):
            node_key = layer + '/Node' + str(num_node)
            node_data_dict = self._get_node_data(node_key)
            self._update_node_output(node_key, node_data_dict)

    def _update_next_layer_inputs(self, current_layer, next_layer):
        next_layer_input = self._get_layer_output(current_layer)
        self._set_layer_input(next_layer, next_layer_input)

    def _update_output_layer(self, child):
        output_layer = child + '/Layer-1'
        num_nodes = self._read_data(output_layer + '/Nodes')
        self._update_current_layer_outputs(output_layer, num_nodes)

    def _set_layer_input(self, layer, layer_input):
        num_nodes = int(self._read_data(layer + '/' + 'Nodes'))
        for nodes in range(0, num_nodes):
            node_key = layer + '/' + 'Node' + str(nodes)
            self._update_node_input(node_key, layer_input)

    def _get_layer_output(self, layer):
        num_nodes = int(self._read_data(layer + '/' + 'Nodes'))
        outputs = np.array([])
        for nodes in range(0, num_nodes):
            read_output = self._read_data(layer + '/' + 'Node' + str(nodes) + '/Output').reshape(1, )
            outputs = np.concatenate((outputs, read_output))

        output_shape = outputs.shape
        outputs = outputs.reshape(output_shape[0], 1)
        return outputs

    def _update_node_input(self, node_key, layer_input):
        layer_input_key = node_key + '/Inputs'
        self._write_data(layer_input_key, layer_input)

    def _update_node_output(self, node_key, node_data_dict):
        layer_output_key = node_key + '/Output'
        layer_output = self._calculate_node_output(node_data_dict)
        self._write_data(layer_output_key, layer_output)

    @staticmethod
    def _write_data(key, data):
        with h5py.File('Children.hdf5', 'r+') as children:
            data_to_write = children[key]['data']
            data_to_write[...] = data

    @staticmethod
    def _read_data(key):
        with h5py.File('Children.hdf5', 'r') as children:
            data = np.array(children.get(key)['data'])

        return data

    @staticmethod
    def _get_child_keys(child):
        keys = []

        def get_keys(name):
            if ('data' in name) and (child in name) and ('Inputs' in name):
                keys.append(name)

        with h5py.File('Children.hdf5', 'r') as children:
            children.visit(get_keys)

        return keys

    def _get_layer_node_data(self, num_child, num_layer, num_node):
        node_dict = dict()
        child = 'Child' + str(num_child) + '/'
        layer = 'Layer' + str(num_layer) + '/'
        node = 'Node' + str(num_node) + '/'

        with h5py.File('Children.hdf5', 'r') as children:
            for chromo in self.neuron_chromo:
                key = child + layer + node + chromo
                node_dict[chromo] = np.array(children.get(key)['data'])

        return node_dict

    @staticmethod
    def _get_layer_node_list(child):

        layer_list = []
        node_list = []

        def _get_list(name):
            if ('data' in name) and (child in name) and ('HiddenLayers' in name) in name:
                layer_list.append(name)

            if ('data' in name) and (child in name) and ('Nodes' in name):
                node_list.append(name)

        with h5py.File('Children.hdf5', 'r') as children:
            children.visit(_get_list)

        return layer_list, node_list

    @staticmethod
    def _get_choice_list(key):
        key_list = []

        def _get_list(name):
            if ('data' not in name) and (key[0] in name) and (key[1] in name) and (key[2] in name):
                key_list.append(name)

        with h5py.File('Children.hdf5', 'r') as children:
            children.visit(_get_list)

        return key_list

    def _get_node_data(self, base_key):
        node_dict = dict()

        with h5py.File('Children.hdf5', 'r') as children:
            for chromo in self.neuron_chromo:
                key = base_key + '/' + chromo
                node_dict[chromo] = np.array(children.get(key)['data'])

        return node_dict

    def _calculate_node_output(self, node):
        node['Weights'], node['Inputs'] = self._ensure_compatibility(node['Weights'], node['Inputs'])
        if node['Enabled'] == 1:
            if node['Recurrent'] == 1:
                recurrent = node['RecurrentWeight'] * node['Output']
            else:
                recurrent = 0
            result = node['Inputs'].transpose().dot(node['Weights']) + node['Bias'] + recurrent
            node_output = self._activation(str(node['Activation']), result)
        else:
            node_output = 0
        return node_output

    def _get_node_output_v2(self, children, base_key):
        if children.get(base_key + '/Enabled')['data'] == 1:
            if children.get(base_key + '/Recurrent')['data'] == 1:
                recurrent = children.get(base_key + '/RecurrentWeight')['data'] * children.get(base_key + '/Output')[
                    'data']
            else:
                recurrent = 0
            result = children.get(base_key + '/Inputs')['data'].transpose().dot(
                children.get(base_key + '/Weights')['data']) + children.get(base_key + '/Bias')['data'] + recurrent
            node_output = self._activation(str(children.get(base_key + '/Activation')['data']), result)
        else:
            node_output = 0
        return node_output

    def _ensure_compatibility(self, weights, inputs):
        new_weights = weights
        new_inputs = inputs
        diff = weights.shape[0] - inputs.shape[0]

        if diff < 0:
            new_weights = self._add_padding(weights, abs(diff))
        elif diff > 0:
            new_inputs = self._add_padding(inputs, abs(diff))

        return new_weights, new_inputs

    # This function will be overwritten by the user class
    def fitness_function(self):
        pass

    # This function will be overwritten by the user class
    def decoder(self):
        pass

    def _activation(self, activation_type, u):
        if activation_type == 'sigmoid':
            return self._sigmoid(u)

        if activation_type == 'relu':
            return self._relu(u)

        if activation_type == 'leakyRelu':
            return self._leaky_relu(u)

        if activation_type == 'tanh':
            return self._tanh(u)

        if activation_type == 'linear':
            return self._linear(u)

    @staticmethod
    def _sigmoid(u):
        try:
            y = 1 / (1 + exp(-u))
        except OverflowError:
            if u > 0:
                y = 1
            else:
                y = 0
        return y

    @staticmethod
    def _relu(u):
        return max(0, u)

    @staticmethod
    def _leaky_relu(u):
        return max(-0.1, u)

    @staticmethod
    def _tanh(u):
        return tanh(u)

    @staticmethod
    def _linear(u):
        return u

    @staticmethod
    def _softmax(u):
        pass

    @staticmethod
    def _add_padding(array, pad_length):
        pad_zeros = np.zeros((pad_length, 1))
        pad_array = np.concatenate((array, pad_zeros))
        return pad_array

    @staticmethod
    def delete_file(file_name):
        remove(file_name)
        print("File Removed!")
