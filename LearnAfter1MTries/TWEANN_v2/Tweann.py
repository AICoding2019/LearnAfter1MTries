import numpy as np
import h5py
from os import remove, rename
from random import choice, random, randint
from math import exp, tanh


class Tweann:
    def __init__(self, num_inputs, num_outputs, population_size=100, crossover_rate=0.7,
                 mutation_rate=0.1, add_node_mutation_rate=0.1, perturbation_rate=0.001,
                 selection_type='roulette_wheel',
                 num_generation=2, log_path=None):

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_type = selection_type
        self.num_generation = num_generation
        self.add_node_mutation_rate = add_node_mutation_rate
        self.perturbation_rate = perturbation_rate
        self.log_path = log_path

        self.fittest_child = None
        self.fittest_child_ever = None
        self.best_fitness_score = -1.0
        self.best_fitness_score_ever = -1.0
        self.total_fitness_score = 0.0
        self.activationFunc = ['sigmoid', 'relu', 'tanh', 'leakyRelu', 'linear']
        self.neuron_chromo = ['Weights', 'Bias', 'Enabled', 'Recurrent', 'RecurrentWeight', 'Activation', 'Inputs',
                              'Output']

    def evolve(self):
        print("Creating population")
        self.create_population()
        print("Population created")
        self.epoch()

    def create_population(self):
        for child in range(0, self.population_size + 2):
            child_init_data = self._create_init_child_chromo('Child' + str(child))
            self.create_child_dataset(child_init_data)

    def _create_init_child_chromo(self, child):
        chromo = dict()
        for output_node_num in range(0, self.num_outputs):
            node_name = 'Node' + str(output_node_num)
            key = child + '/Layer-1/' + node_name
            chromo = self._new_node_dict(key, self.num_inputs)

        chromo[child + '/Layer-1/Nodes'] = self.num_outputs
        chromo[child + '/HiddenLayers'] = 0
        chromo[child + '/Fitness'] = 0.0
        return chromo

    @staticmethod
    def create_child_dataset(child_dataset, hdf5_file='Children.hdf5'):
        with h5py.File(hdf5_file, 'a') as children:
            for key in child_dataset:
                group = children.create_group(key)
                group.create_dataset('data', data=child_dataset[key])

    # Todo
    @staticmethod
    def write_to_child_dataset(child_dataset, hdf5_file='Children.hdf5'):
        with h5py.File(hdf5_file, 'r+') as children:
            for key in child_dataset:
                children.create_dataset('data', data=child_dataset[key])

    def epoch(self):
        generation = 0
        while generation < self.num_generation:
            generation += 1
            self._update_children_fitness()
            self._generate_babies()
            self._create_new_population()

    def _update_children_fitness(self):
        child = None
        self.best_fitness_score = -1
        self.total_fitness_score = 0

        for num_child in range(0, self.population_size + 2):
            child = 'Child' + str(num_child)
            fitness, info = self.fitness_function(child)
            self._update_fitness(child, fitness)

        self._update_best_child(child)

    def _update_fitness(self, child, fitness):
        key = child + '/Fitness'
        self._write_data(key, fitness)
        self.total_fitness_score += fitness

        if fitness > self.best_fitness_score:
            self.best_fitness_score = fitness
            self.fittest_child = child

    def _update_best_child(self, child):
        if self.best_fitness_score > self.best_fitness_score_ever:
            self.fittest_child_ever = child
            self.best_fitness_score_ever = self.best_fitness_score

    def _generate_babies(self):
        new_babies = 0
        baby_num = 0

        while new_babies < self.population_size:
            mum_selected, dad_selected = self._parent_selection()
            self._crossover_babies(mum_selected, dad_selected, baby_num)
            self._mutate_babies(baby_num)
            baby_num += 2
            new_babies = baby_num

    def _parent_selection(self):
        mum_selected = self._selection()
        dad_selected = self._selection()
        return mum_selected, dad_selected

    def _crossover_babies(self, mum_selected, dad_selected, baby_num):
        if random() < self.crossover_rate:
            self._crossover(mum_selected, dad_selected, baby_num)

    def _mutate_babies(self, baby_num):
        self._mutate(baby_num)
        self._mutate(baby_num + 1)

    def _selection(self):
        if self.selection_type == 'rouletteWheel':
            return self._roulette_wheel_selection()

    def _roulette_wheel_selection(self):
        roulette_slice = random()
        total = 0.0
        selected_child = 'Child' + str(self.population_size + 2)

        for num_child in range(0, self.population_size + 2):
            child = 'Child' + str(num_child)
            key = child + '/Fitness'
            fitness = self._read_data(key)
            total += fitness / self.total_fitness_score

            if total >= roulette_slice:
                selected_child = child
                break

        return selected_child

    def _crossover(self, mum, dad, baby_num):
        layer, node, node_key = self._crossover_node_pick(mum, dad)
        self._clone(mum, baby_num)
        self._clone(dad, baby_num + 1)

        if self._check_node_exist(mum, layer, node) and self._check_node_exist(dad, layer, node):
            self._crossover_node(mum, dad, node_key, baby_num)

    @staticmethod
    def _clone(parent, baby_num):
        with h5py.File('Children.hdf5', 'a') as children:
            with h5py.File('Babies.hdf5', 'a') as babies:
                children.copy(parent, babies, name='Child' + str(baby_num))

    # Todo Nice idea not used
    @staticmethod
    def _inherit(recipient, donor, node_key, baby_num):
        with h5py.File('Children.hdf5', 'a') as children:
            with h5py.File('Babies.hdf5', 'a') as babies:
                children.copy(recipient, babies, name='Child' + str(baby_num))
                node_to_inherit = donor + '/' + node_key
                children.copy(node_to_inherit, babies, name='Child' + str(baby_num) + '/' + node_key)

    def _crossover_node(self, mum, dad, node_key, baby_num):
        mum_dict, dad_dict = self._get_crossover_node_dict(mum, dad, node_key)
        baby1_dict, baby2_dict = self._crossover_node_element(mum_dict, dad_dict)
        baby1_node_key = 'Child' + str(baby_num - 1) + '/' + node_key
        baby2_node_key = 'Child' + str(baby_num) + '/' + node_key
        self._set_node_data(baby1_node_key, baby1_dict)
        self._set_node_data(baby2_node_key, baby2_dict)

    def _get_crossover_node_dict(self, mum, dad, node_key):
        mum_dict = self._get_node_data(mum + '/' + node_key)
        dad_dict = self._get_node_data(dad + '/' + node_key)
        return mum_dict, dad_dict

    def _crossover_node_element(self, mum_dict, dad_dict):
        baby1_dict = dict()
        baby2_dict = dict()

        for element in mum_dict:
            if isinstance(mum_dict[element], str):
                baby1_dict[element], baby2_dict[element] = self._cross_over_swap(mum_dict[element], dad_dict[element])
            if isinstance(mum_dict[element], np.ndarray):
                baby1_dict[element], baby2_dict[element] = self._cross_over_random_swap_point(mum_dict[element],
                                                                                              dad_dict[element])
            if isinstance(mum_dict[element], int):
                baby1_dict[element], baby2_dict[element] = self._cross_over_swap(mum_dict[element], dad_dict[element])

        return baby1_dict, baby2_dict

    @staticmethod
    def _cross_over_swap(mum_element, dad_element, rate=0.5):
        if random() < rate:
            baby1_element = mum_element
            baby2_element = dad_element

        else:
            baby1_element = dad_element
            baby2_element = mum_element

        return baby1_element, baby2_element

    def _cross_over_random_swap_point(self, mum_array, dad_array):
        max_swap = self._get_max_swap_point(mum_array, dad_array)

        baby1_array = mum_array
        baby2_array = dad_array

        if max_swap != 0:
            swap_point = randint(0, max_swap)
            baby1_array, baby2_array = self._swap_array(mum_array, dad_array, max_swap, swap_point)

        return baby1_array, baby2_array

    def _crossover_node_pick(self, mum, dad):
        dominant = choice([mum, dad])
        layer, node, node_key = self._pick_layer_node(dominant)

        return layer, node, node_key

    def _pick(self, key):
        num_key = int(self._read_data(key))
        pick = randint(0, num_key)
        return pick

    def _pick_layer_node(self, child):
        pick_layer = self._pick(child + '/HiddenLayers')
        layer = pick_layer - 1
        layer_key = 'Layer' + str(layer)
        node = self._pick(child + '/' + layer_key + '/Nodes')
        node_key = layer_key + '/Node' + str(node)

        return layer, node, node_key

    def _check_node_exist(self, parent, layer, node):
        num_hidden_layers = int(self._read_data(parent + '/HiddenLayers'))

        if layer > num_hidden_layers:
            return False
        else:
            num_nodes = int(self._read_data(parent + '/Layer' + str(layer) + '/Nodes'))
            if node > num_nodes:
                return False
            else:
                return True

    def _mutate(self, child_num):
        if random() < self.mutation_rate:
            self._mutate_node(child_num)
        if random() < self.add_node_mutation_rate:
            self._mutate_by_adding_node(child_num)

    def _mutate_node(self, child_num):
        node_key = self._mutate_node_pick(child_num)
        child_dict = self._get_node_data(node_key)
        mutate_child_dict = self._mutate_node_element(child_dict)
        self._set_node_data(node_key, mutate_child_dict)

    def _mutate_node_pick(self, child_num):
        child = 'Child' + str(child_num)
        _, _, node_key = self._pick_layer_node(child)
        child_node_key = child + '/' + node_key
        return child_node_key

    def _mutate_node_element(self, child_dict):
        baby_dict = dict()

        for element in child_dict:
            if isinstance(child_dict[element], str):
                baby_dict[element] = choice(self.activationFunc)
            if isinstance(child_dict[element], np.ndarray):
                baby_dict[element] = self._mutate_array(child_dict[element], self.perturbation_rate)
            if isinstance(child_dict[element], int):
                baby_dict[element] = choice([0, 1])

        return baby_dict

    @staticmethod
    def _mutate_array(array, perturbation_rate):
        mutate_array = array.copy()
        shape_array = array.shape
        pick_array_element = random(shape_array[0])
        mutate_element = array[pick_array_element] + (-perturbation_rate + 2 * perturbation_rate * random())
        mutate_array[pick_array_element] = mutate_element
        return mutate_array

    def _mutate_by_adding_node(self, child_num):
        child = 'Child' + str(child_num)
        num_hidden_layers = int(self._read_data(child + '/HiddenLayers'))
        layer = randint(0, num_hidden_layers + 1)
        if layer > num_hidden_layers:
            self._increase_child_layer(child, layer)
        else:
            self._increase_child_layer_node(child, layer)

    def _increase_child_layer(self, child, layer):
        self._write_data(child + '/HiddenLayers', layer, hdf5_file='Babies.hdf5')
        self._increase_child_layer_node(child, layer)

    def _increase_child_layer_node(self, child, layer):
        self._write_data(child + '/Layer' + str(layer) + '/Nodes', 1, hdf5_file='Babies.hdf5')
        num_inputs = int(self._read_data(child + '/Layer' + str(layer) + 'Inputs'))
        key = child + '/Layer' + str(layer) + '/Node0'
        self._add_new_node(key, num_inputs)

    def _add_new_node(self, key, num_inputs, hdf5_file='Babies.hdf5'):
        node_dict = self._new_node_dict(key, num_inputs)
        self.create_child_dataset(node_dict, hdf5_file=hdf5_file)
        # self._set_node_data(key, node_dict, hdf5_file=hdf5_file)

    def _new_node_dict(self, key, num_inputs):
        chromo = dict()
        chromo[key + '/Weights'] = np.random.rand(num_inputs, 1)
        chromo[key + '/Bias'] = np.random.rand(1, 1)
        chromo[key + '/Enabled'] = 1
        chromo[key + '/Recurrent'] = choice([0, 1])
        chromo[key + '/RecurrentWeight'] = np.random.rand(1, 1)
        chromo[key + '/Activation'] = choice(self.activationFunc)
        chromo[key + '/Inputs'] = np.zeros((num_inputs, 1))
        chromo[key + '/Output'] = np.zeros((1, 1))

        return chromo

    def _create_new_population(self):
        self._add_best_children()
        self.delete_file('Children.hdf5')
        rename('Babies.hdf5', 'Children.hdf5')

    def _add_best_children(self):
        best_baby_num = self.population_size + 1
        self._clone(self.fittest_child, best_baby_num)
        best_baby_ever_num = self.population_size + 2
        self._clone(self.fittest_child_ever, best_baby_ever_num)

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
    def _write_data(key, data, hdf5_file='Children.hdf5'):
        with h5py.File(hdf5_file, 'r+') as children:
            data_to_write = children[key]['data']
            data_to_write[...] = data

    @staticmethod
    def _read_data(key, hdf5_file='Children.hdf5'):
        with h5py.File(hdf5_file, 'r') as children:
            data = np.array(children.get(key)['data'])

        return data

    @staticmethod
    def _get_child_keys(child, hdf5_file='Children.hdf5'):
        keys = []

        def get_keys(name):
            if ('data' in name) and (child in name) and ('Inputs' in name):
                keys.append(name)

        with h5py.File(hdf5_file, 'r') as children:
            children.visit(get_keys)

        return keys

    def _get_layer_node_data(self, num_child, num_layer, num_node, hdf5_file='Children.hdf5'):
        node_dict = dict()
        child = 'Child' + str(num_child) + '/'
        layer = 'Layer' + str(num_layer) + '/'
        node = 'Node' + str(num_node) + '/'

        with h5py.File(hdf5_file, 'r') as children:
            for chromo in self.neuron_chromo:
                key = child + layer + node + chromo
                node_dict[chromo] = np.array(children.get(key)['data'])

        return node_dict

    @staticmethod
    def _get_layer_node_list(child, hdf5_file='Children.hdf5'):

        layer_list = []
        node_list = []

        def _get_list(name):
            if ('data' in name) and (child in name) and ('HiddenLayers' in name) in name:
                layer_list.append(name)

            if ('data' in name) and (child in name) and ('Nodes' in name):
                node_list.append(name)

        with h5py.File(hdf5_file, 'r') as children:
            children.visit(_get_list)

        return layer_list, node_list

    @staticmethod
    def _get_choice_list(key, hdf5_file='Children.hdf5'):
        key_list = []

        def _get_list(name):
            if ('data' not in name) and (key[0] in name) and (key[1] in name) and (key[2] in name):
                key_list.append(name)

        with h5py.File(hdf5_file, 'r') as children:
            children.visit(_get_list)

        return key_list

    def _get_node_data(self, base_key, hdf5_file='Children.hdf5'):
        node_dict = dict()

        with h5py.File(hdf5_file, 'r') as children:
            for chromo in self.neuron_chromo:
                key = base_key + '/' + chromo
                node_dict[chromo] = np.array(children.get(key)['data'])

        return node_dict

    def _set_node_data(self, node_key, node_dict, hdf5_file='Babies.hdf5'):
        for chromo in self.neuron_chromo:
            key = node_key + '/' + chromo
            data = node_dict[chromo]
            self._write_data(key, data, hdf5_file=hdf5_file)

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

    @staticmethod
    def _get_max_swap_point(mum_array, dad_array):
        mum_shape = mum_array.shape[0]
        dad_shape = dad_array.shape[0]

        max_swap = min(mum_shape, dad_shape)
        return max_swap

    @staticmethod
    def _swap_array(mum_array, dad_array, max_swap, swap_point):
        baby1_array = mum_array.copy()
        baby2_array = dad_array.copy()

        baby1_array[0:swap_point] = mum_array[0:swap_point]
        baby1_array[swap_point:max_swap] = dad_array[swap_point:max_swap]

        baby2_array[0:swap_point] = dad_array[0:swap_point]
        baby2_array[swap_point:max_swap] = mum_array[swap_point:max_swap]

        return baby1_array, baby2_array

    # This function will be overwritten by the user class
    def fitness_function(self, child):
        return 0

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
