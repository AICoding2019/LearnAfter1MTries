import unittest
from LearnAfter1MTries.TWEANN_v2.Tweann import *
from os import path
from math import fabs


class TweannTestCase(unittest.TestCase):

    def setUp(self):
        self.num_inputs = 2
        self.num_outputs = 1

        self.test_tweann = Tweann(self.num_inputs, self.num_outputs, population_size=5)
        self.test_tweann_2 = Tweann(self.num_inputs, 2, population_size=5)
        self.activationFunc = self.test_tweann.activationFunc
        self.test_child0 = {'Child0/Layer-1/Node0/Weights': np.random.rand(self.num_inputs, 1),
                            'Child0/Layer-1/Node0/Bias': np.random.rand(1, 1),
                            'Child0/Layer-1/Node0/Enabled': 1,
                            'Child0/Layer-1/Node0/Recurrent': choice([0, 1]),
                            'Child0/Layer-1/Node0/RecurrentWeight': np.random.rand(1, 1),
                            'Child0/Layer-1/Node0/Activation': choice(self.activationFunc),
                            'Child0/Layer-1/Node0/Inputs': np.zeros((self.num_inputs, 1)),
                            'Child0/Layer-1/Node0/Output': np.zeros((1, 1)),
                            'Child0/Layer-1/Nodes': 1,
                            'Child0/HiddenLayers': 0,
                            'Child0/Fitness': 0.0,
                            }

        self.test_child1 = {'Child0/Layer0/Node0/Weights': np.random.rand(self.num_inputs, 1),
                            'Child0/Layer0/Node0/Bias': np.random.rand(1, 1),
                            'Child0/Layer0/Node0/Enabled': choice([0, 1]),
                            'Child0/Layer0/Node0/Recurrent': choice([0, 1]),
                            'Child0/Layer0/Node0/RecurrentWeight': np.random.rand(1, 1),
                            'Child0/Layer0/Node0/Activation': choice(self.activationFunc),
                            'Child0/Layer0/Node0/Inputs': np.zeros((self.num_inputs, 1)),
                            'Child0/Layer0/Node0/Output': np.zeros((1, 1)),
                            'Child0/Layer0/Node1/Weights': np.random.rand(self.num_inputs, 1),
                            'Child0/Layer0/Node1/Bias': np.random.rand(1, 1),
                            'Child0/Layer0/Node1/Enabled': choice([0, 1]),
                            'Child0/Layer0/Node1/Recurrent': choice([0, 1]),
                            'Child0/Layer0/Node1/RecurrentWeight': np.random.rand(1, 1),
                            'Child0/Layer0/Node1/Activation': choice(self.activationFunc),
                            'Child0/Layer0/Node1/Inputs': np.zeros((self.num_inputs, 1)),
                            'Child0/Layer0/Node1/Output': np.zeros((1, 1)),
                            'Child0/Layer0/Nodes': 2,
                            'Child0/Layer-1/Node0/Weights': np.random.rand(self.num_inputs, 1),
                            'Child0/Layer-1/Node0/Bias': np.random.rand(1, 1),
                            'Child0/Layer-1/Node0/Enabled': 1,
                            'Child0/Layer-1/Node0/Recurrent': choice([0, 1]),
                            'Child0/Layer-1/Node0/RecurrentWeight': np.random.rand(1, 1),
                            'Child0/Layer-1/Node0/Activation': choice(self.activationFunc),
                            'Child0/Layer-1/Node0/Inputs': np.zeros((1, 1)),
                            'Child0/Layer-1/Node0/Output': np.zeros((1, 1)),
                            'Child0/Layer-1/Node1/Weights': np.random.rand(self.num_inputs, 1),
                            'Child0/Layer-1/Node1/Bias': np.random.rand(1, 1),
                            'Child0/Layer-1/Node1/Enabled': 1,
                            'Child0/Layer-1/Node1/Recurrent': choice([0, 1]),
                            'Child0/Layer-1/Node1/RecurrentWeight': np.random.rand(self.num_outputs, 1),
                            'Child0/Layer-1/Node1/Activation': choice(self.activationFunc),
                            'Child0/Layer-1/Node1/Inputs': np.zeros((self.num_inputs, 1)),
                            'Child0/Layer-1/Node1/Output': np.zeros((1, 1)),
                            'Child0/Layer-1/Nodes': 2,
                            'Child0/HiddenLayers': 1,
                            'Child0/Fitness': 0.0
                            }

        self.test_child2 = {'Child0/Layer0/Node0/Weights': np.array([[0.90227926], [0.20423683]]),
                            'Child0/Layer0/Node0/Bias': np.array([[0.77573306]]),
                            'Child0/Layer0/Node0/Enabled': 1,
                            'Child0/Layer0/Node0/Recurrent': 1,
                            'Child0/Layer0/Node0/RecurrentWeight': np.array([[0.5]]),
                            'Child0/Layer0/Node0/Activation': 'sigmoid',
                            'Child0/Layer0/Node0/Inputs': np.array([[1.0], [1.0]]),
                            'Child0/Layer0/Node0/Output': np.array([[0.2]]),
                            'Child0/Layer0/Node1/Weights': np.array([[0.90227926], [0.20423683]]),
                            'Child0/Layer0/Node1/Bias': np.array([[0.77573306]]),
                            'Child0/Layer0/Node1/Enabled': 1,
                            'Child0/Layer0/Node1/Recurrent': 1,
                            'Child0/Layer0/Node1/RecurrentWeight': np.array([[0.5]]),
                            'Child0/Layer0/Node1/Activation': 'sigmoid',
                            'Child0/Layer0/Node1/Inputs': np.array([[1.0], [1.0]]),
                            'Child0/Layer0/Node1/Output': np.array([[0.2]]),
                            'Child0/Layer0/Nodes': 2,
                            'Child0/Layer1/Node0/Weights': np.array([[0.90227926], [0.20423683]]),
                            'Child0/Layer1/Node0/Bias': np.array([[0.77573306]]),
                            'Child0/Layer1/Node0/Enabled': 1,
                            'Child0/Layer1/Node0/Recurrent': 1,
                            'Child0/Layer1/Node0/RecurrentWeight': np.array([[0.5]]),
                            'Child0/Layer1/Node0/Activation': 'sigmoid',
                            'Child0/Layer1/Node0/Inputs': np.array([[2.0], [2.0]]),
                            'Child0/Layer1/Node0/Output': np.zeros((1, 1)),
                            'Child0/Layer1/Node1/Weights': np.array([[0.90227926], [0.20423683]]),
                            'Child0/Layer1/Node1/Bias': np.array([[0.77573306]]),
                            'Child0/Layer1/Node1/Enabled': 1,
                            'Child0/Layer1/Node1/Recurrent': 1,
                            'Child0/Layer1/Node1/RecurrentWeight': np.array([[0.5]]),
                            'Child0/Layer1/Node1/Activation': 'sigmoid',
                            'Child0/Layer1/Node1/Inputs': np.array([[2.0], [2.0]]),
                            'Child0/Layer1/Node1/Output': np.array([[1]]),
                            'Child0/Layer1/Nodes': 2,
                            'Child0/Layer-1/Node0/Weights': np.array([[0.90227926], [0.20423683]]),
                            'Child0/Layer-1/Node0/Bias': np.array([[0.77573306]]),
                            'Child0/Layer-1/Node0/Enabled': 1,
                            'Child0/Layer-1/Node0/Recurrent': 1,
                            'Child0/Layer-1/Node0/RecurrentWeight': np.array([[0.5]]),
                            'Child0/Layer-1/Node0/Activation': 'sigmoid',
                            'Child0/Layer-1/Node0/Inputs': np.array([[1.0], [1.0]]),
                            'Child0/Layer-1/Node0/Output': np.array([[0.2]]),
                            'Child0/Layer-1/Nodes': 1,
                            'Child0/HiddenLayers': 2,
                            'Child0/Fitness': 0.0
                            }
        self.test_child3 = {'Child1/Layer0/Node0/Weights': np.random.rand(self.num_inputs, 1),
                            'Child1/Layer0/Node0/Bias': np.random.rand(1, 1),
                            'Child1/Layer0/Node0/Enabled': choice([0, 1]),
                            'Child1/Layer0/Node0/Recurrent': choice([0, 1]),
                            'Child1/Layer0/Node0/RecurrentWeight': np.random.rand(1, 1),
                            'Child1/Layer0/Node0/Activation': choice(self.activationFunc),
                            'Child1/Layer0/Node0/Inputs': np.zeros((self.num_inputs, 1)),
                            'Child1/Layer0/Node0/Output': np.zeros((1, 1)),
                            'Child1/Layer0/Node1/Weights': np.random.rand(self.num_inputs, 1),
                            'Child1/Layer0/Node1/Bias': np.random.rand(1, 1),
                            'Child1/Layer0/Node1/Enabled': choice([0, 1]),
                            'Child1/Layer0/Node1/Recurrent': choice([0, 1]),
                            'Child1/Layer0/Node1/RecurrentWeight': np.random.rand(1, 1),
                            'Child1/Layer0/Node1/Activation': choice(self.activationFunc),
                            'Child1/Layer0/Node1/Inputs': np.zeros((self.num_inputs, 1)),
                            'Child1/Layer0/Node1/Output': np.zeros((1, 1)),
                            'Child1/Layer0/Nodes': 2,
                            'Child1/Layer-1/Node0/Weights': np.random.rand(self.num_inputs, 1),
                            'Child1/Layer-1/Node0/Bias': np.random.rand(1, 1),
                            'Child1/Layer-1/Node0/Enabled': 1,
                            'Child1/Layer-1/Node0/Recurrent': choice([0, 1]),
                            'Child1/Layer-1/Node0/RecurrentWeight': np.random.rand(1, 1),
                            'Child1/Layer-1/Node0/Activation': choice(self.activationFunc),
                            'Child1/Layer-1/Node0/Inputs': np.zeros((1, 1)),
                            'Child1/Layer-1/Node0/Output': np.zeros((1, 1)),
                            'Child1/Layer-1/Node1/Weights': np.random.rand(self.num_inputs, 1),
                            'Child1/Layer-1/Node1/Bias': np.random.rand(1, 1),
                            'Child1/Layer-1/Node1/Enabled': 1,
                            'Child1/Layer-1/Node1/Recurrent': choice([0, 1]),
                            'Child1/Layer-1/Node1/RecurrentWeight': np.random.rand(self.num_outputs, 1),
                            'Child1/Layer-1/Node1/Activation': choice(self.activationFunc),
                            'Child1/Layer-1/Node1/Inputs': np.zeros((self.num_inputs, 1)),
                            'Child1/Layer-1/Node1/Output': np.zeros((1, 1)),
                            'Child1/Layer-1/Nodes': 2,
                            'Child1/HiddenLayers': 1,
                            'Child1/Fitness': 0.0
                            }

        class TestTweannClassUsage(Tweann):
            def fitness_function(self, child):
                X = [[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]]
                y = [0, 1, 1, 0]
                total_error = 0
                num_correct = 0
                prediction = []
                prediction_bin = []
                for index in range(0, len(X)):
                    nn_input = np.array([X[index]]).reshape(2, 1)
                    nn_output = TestTweannClassUsage.get_neural_network_output(self, child, nn_input)
                    total_error += fabs(y[index] - nn_output)
                    if nn_output < 0.5:
                        predict = 0
                    else:
                        predict = 1

                    if y[index] == predict:
                        num_correct += 1

                    prediction.append(nn_output)
                    prediction_bin.append(predict)
                fitness = (num_correct / 4 + (1 / (1 + total_error))) / 2
                return fitness, [num_correct, prediction_bin]

        self.test_tweannClassUsage = TestTweannClassUsage(self.num_inputs, self.num_outputs, population_size=10)

    def tearDown(self):
        if path.exists('Children.hdf5'):
            remove('Children.hdf5')

        if path.exists('Babies.hdf5'):
            remove('Babies.hdf5')

    def test_create_init_child_chromo_keys(self):
        actual = self.test_tweann._create_init_child_chromo('Child0').keys()
        expected = self.test_child0.keys()
        self.assertEqual(expected, actual)

    def test_create_init_child_chromo_dataTypes(self):
        child = self.test_tweann_2._create_init_child_chromo('Child0')
        actual = []
        for item in child:
            if isinstance(child[item], str):
                actual.append(True)
            if isinstance(child[item], np.ndarray):
                actual.append(child[item].shape)
            if isinstance(child[item], int):
                actual.append(True)
            if isinstance(child[item], float):
                actual.append(True)
            if isinstance(child[item], type(None)):
                actual.append(None)

        expected = [(self.num_inputs, 1), (1, 1), True, True, (self.num_outputs, 1), True, (2, 1),
                    (self.num_outputs, 1), True, True, True]
        self.assertEqual(expected, actual)

    def test_create_child0_test(self):
        test_child = self.test_child0
        self.test_tweann.create_child_dataset(test_child)
        expected = [np.array(test_child[key]).tolist() for key in test_child]

        actual = []
        with h5py.File('Children.hdf5', 'r') as children:
            for key in test_child:
                data = children.get(key)['data']
                actual.append(np.array(data).tolist())
        self.assertEqual(expected, actual)

    def test_create_child1_test(self):
        test_child = self.test_child1
        self.test_tweann.create_child_dataset(test_child)
        expected = [np.array(test_child[key]).tolist() for key in test_child]

        actual = []
        with h5py.File('Children.hdf5', 'r') as children:
            for key in test_child:
                data = children.get(key)['data']
                actual.append(np.array(data).tolist())
        self.assertEqual(expected, actual)

    def test_create_child2_test(self):
        test_child = self.test_child2
        self.test_tweann.create_child_dataset(test_child)
        expected = [np.array(test_child[key]).tolist() for key in test_child]

        actual = []
        with h5py.File('Children.hdf5', 'r') as children:
            for key in test_child:
                data = children.get(key)['data']
                actual.append(np.array(data).tolist())
        self.assertEqual(expected, actual)

    def test_create_population(self):
        self.test_tweann.create_population()
        expected = []
        actual = []

        def get_keys(name):
            if 'data' in name:
                actual.append(name)

        with h5py.File('Children.hdf5', 'r') as children:
            children.visit(get_keys)

        for index in range(0, 7):
            child_replacement = 'Child' + str(index)
            for key in self.test_child0:
                new_child = key.replace('Child0', child_replacement)
                new_child_end = new_child + '/data'
                expected.append(new_child_end)
        expected.sort()
        actual.sort()

        self.assertEqual(expected, actual)

    def test_calculate_node_output(self):
        node = {'Weights': np.array([[0.90227926], [0.20423683]]),
                'Bias': np.array([[0.77573306]]),
                'Enabled': 1,
                'Recurrent': 1,
                'RecurrentWeight': np.array([[0.5]]),
                'Activation': 'sigmoid',
                'Inputs': np.array([[1], [1]]),
                'Output': np.array([[0.2]])
                }
        actual = self.test_tweann._calculate_node_output(node)
        expected = self.test_tweann._sigmoid(1.98224915)

        self.assertEqual(expected, actual)

    def test_get_layer_node_data(self):
        self.test_tweann.create_population()

        num_child = 0
        num_layer = -1
        num_node = 0

        actual = self.test_tweann._get_layer_node_data(num_child, num_layer, num_node)
        expected = actual
        self.assertEqual(expected, actual)

    def test_get_choice_list(self):
        self.test_tweann.create_child_dataset(self.test_child2)
        key = ['Child0', '', 'Node']
        actual = self.test_tweann._get_choice_list(key)
        expected = actual
        self.assertEqual(expected, actual)

    def test_read_data(self):
        self.test_tweann.create_population()
        key = 'Child0/Layer-1/Node0/Output'
        actual = self.test_tweann._read_data(key)
        expected = np.zeros((1, 1))
        self.assertEqual(expected, actual)

    def test_write_data(self):
        self.test_tweann.create_population()

        key = 'Child0/Layer-1/Node0/Inputs'
        expected = np.array([[100], [100]])
        self.test_tweann._write_data(key, expected)
        actual = self.test_tweann._read_data(key)
        self.assertEqual(expected.tolist(), actual.tolist())

    def test_update_node_input(self):
        self.test_tweann.create_population()
        node_key = 'Child0/Layer-1/Node0'
        expected = np.array([[1000], [1000]])
        self.test_tweann._update_node_input(node_key, expected)
        actual = self.test_tweann._read_data(node_key + '/' + 'Inputs')
        self.assertEqual(expected.tolist(), actual.tolist())

    def test_update_node_output(self):
        self.test_tweann.create_population()
        node_key = 'Child0/Layer-1/Node0'
        node_data_dict = {'Weights': np.array([[0.90227926], [0.20423683]]),
                          'Bias': np.array([[0.77573306]]),
                          'Enabled': 1,
                          'Recurrent': 1,
                          'RecurrentWeight': np.array([[0.5]]),
                          'Activation': 'sigmoid',
                          'Inputs': np.array([[1], [1]]),
                          'Output': np.array([[0.2]])
                          }

        self.test_tweann._update_node_output(node_key, node_data_dict)
        expected = self.test_tweann._sigmoid(1.98224915)

        key = node_key + '/' + 'Output'
        actual = self.test_tweann._read_data(key)
        self.assertEqual(expected, actual)

    def test_update_input_layer_no_hiddenLayers(self):
        test_child = self.test_child0
        child = 'Child0'
        key = child + '/' + 'HiddenLayers'
        self.test_tweann.create_child_dataset(test_child)
        num_hidden_layers = self.test_tweann._read_data(key)
        nn_input = np.array([[100], [100]])
        self.test_tweann._update_input_layer(child, num_hidden_layers, nn_input)
        expected = [nn_input.tolist()]
        node0_key = 'Child0/Layer-1/Node0/Inputs'
        actual_node0 = self.test_tweann._read_data(node0_key).tolist()
        actual = [actual_node0]
        self.assertEqual(expected, actual)

    def test_update_input_layer_hiddenLayers(self):
        test_child = self.test_child1
        child = 'Child0'
        key = child + '/' + 'HiddenLayers'
        self.test_tweann.create_child_dataset(test_child)
        num_hidden_layers = self.test_tweann._read_data(key)
        nn_input = np.array([[100], [100]])
        self.test_tweann._update_input_layer(child, num_hidden_layers, nn_input)
        expected = [nn_input.tolist(), nn_input.tolist()]
        node0_key = 'Child0/Layer0/Node0/Inputs'
        node1_key = 'Child0/Layer0/Node1/Inputs'
        actual_node0 = self.test_tweann._read_data(node0_key).tolist()
        actual_node1 = self.test_tweann._read_data(node1_key).tolist()
        actual = [actual_node0, actual_node1]
        self.assertEqual(expected, actual)

    def test_update_hidden_layers(self):
        test_child = self.test_child1
        child = 'Child0'
        key = child + '/' + 'HiddenLayers'
        self.test_tweann.create_child_dataset(test_child)
        num_hidden_layers = self.test_tweann._read_data(key)
        self.test_tweann._update_hidden_layers(child, num_hidden_layers)

    def test_update_output_layer(self):
        test_child = self.test_child2
        child = 'Child0'
        self.test_tweann.create_child_dataset(test_child)
        node0_input_key = 'Child0/Layer-1/Node0/Output'
        self.test_tweann._update_output_layer(child)
        actual = self.test_tweann._read_data(node0_input_key)
        expected = self.test_tweann._sigmoid(1.98224915)
        self.assertEqual(expected, actual)

    def test_update_next_layer_inputs(self):
        test_child = self.test_child2
        self.test_tweann.create_child_dataset(test_child)
        current_layer = 'Child0/Layer0'
        next_layer = 'Child0/Layer1'

        self.test_tweann._update_next_layer_inputs(current_layer, next_layer)
        expected = [[0.2], [0.2]]
        actual = self.test_tweann._read_data(next_layer + '/Node0/Inputs').tolist()
        self.assertEqual(expected, actual)

    def test_update_current_layer_outputs(self):
        test_child = self.test_child2
        self.test_tweann.create_child_dataset(test_child)
        layer = 'Child0/Layer0'
        num_nodes = self.test_tweann._read_data(layer + '/Nodes')
        self.test_tweann._update_current_layer_outputs(layer, num_nodes)
        actual_node0_output = self.test_tweann._read_data(layer + '/Node0/Output').tolist()
        actual_node1_output = self.test_tweann._read_data(layer + '/Node1/Output').tolist()
        actual = [actual_node0_output, actual_node1_output]
        expected_calc = self.test_tweann._sigmoid(1.98224915)
        expected = [[[expected_calc]], [[expected_calc]]]

        self.assertEqual(expected, actual)

    def test_update_neural_network(self):
        test_child = self.test_child2
        child = 'Child0'
        nn_input = np.array([[1], [1]])
        self.test_tweann.create_child_dataset(test_child)
        self.test_tweann._update_neural_network(child, nn_input)
        key = 'Child0/Layer-1/Node0/Output'
        actual = self.test_tweann._read_data(key).tolist()

        expected = [[self.test_tweann._sigmoid(1.98224915)]]
        self.assertEqual(expected, actual)

    def test_get_layer_output(self):
        test_child = self.test_child2
        output_layer = 'Child0/Layer0'
        self.test_tweann.create_child_dataset(test_child)
        actual = self.test_tweann._get_layer_output(output_layer).tolist()
        expected = np.array([[0.2], [0.2]]).tolist()

        self.assertEqual(expected, actual)

    def test_get_neural_network_output(self):
        test_child = self.test_child2
        child = 'Child0'
        nn_input = np.array([[1], [1]])
        self.test_tweann.create_child_dataset(test_child)
        self.test_tweann.get_neural_network_output(child, nn_input)
        key = 'Child0/Layer-1/Node0/Output'
        actual = self.test_tweann._read_data(key).tolist()

        expected = [[self.test_tweann._sigmoid(1.98224915)]]
        self.assertEqual(expected, actual)

    def test_add_padding(self):
        array = np.array([[1], [1]])
        pad_length = 4
        actual = self.test_tweann._add_padding(array, pad_length).tolist()
        expected = [[1], [1], [0], [0], [0], [0]]
        self.assertEqual(expected, actual)

    def test_ensure_compatibility_weights_equal_inputs(self):
        weights = np.array([[1], [1]])
        inputs = np.array([[2], [2]])
        new_weights, new_inputs = self.test_tweann._ensure_compatibility(weights, inputs)
        actual = [new_weights.tolist(), new_inputs.tolist()]
        expected = [[[1], [1]], [[2], [2]]]
        self.assertEqual(expected, actual)

    def test_ensure_compatibility_weights_greater_inputs(self):
        weights = np.array([[1], [1], [1], [1]])
        inputs = np.array([[2], [2]])
        new_weights, new_inputs = self.test_tweann._ensure_compatibility(weights, inputs)
        actual = [new_weights.tolist(), new_inputs.tolist()]
        expected = [[[1], [1], [1], [1]], [[2], [2], [0], [0]]]
        self.assertEqual(expected, actual)

    def test_ensure_compatibility_weights_less_inputs(self):
        weights = np.array([[1], [1]])
        inputs = np.array([[2], [2], [2], [2]])
        new_weights, new_inputs = self.test_tweann._ensure_compatibility(weights, inputs)
        actual = [new_weights.tolist(), new_inputs.tolist()]
        expected = [[[1], [1], [0], [0]], [[2], [2], [2], [2]]]
        self.assertEqual(expected, actual)

    def test_fitness_function(self):
        test_child = self.test_child2
        self.test_tweannClassUsage.create_child_dataset(test_child)
        child = 'Child0'
        actual = self.test_tweannClassUsage.fitness_function(child)
        expected = (0.41844584296725107,
                    [2,
                     [1, 1, 1, 1]])

        self.assertEqual(expected, actual)

    def test_update_fitness(self):
        test_child = self.test_child0
        self.test_tweann.create_child_dataset(test_child)
        child = 'Child0'
        fitness = 1000.0
        self.test_tweann._update_fitness(child, fitness)
        actual = self.test_tweann._read_data(child + '/Fitness')
        expected = fitness
        self.assertEqual(expected, actual)

    def test_update_best_child(self):
        child = 'Child0'
        self.test_tweann.fittest_child = 'Child5'
        self.test_tweann.fittest_child_ever = 'Child10'
        self.test_tweann.best_fitness_score = 10
        self.test_tweann.best_fitness_score_ever = 1

        self.test_tweann.create_population()
        self.test_tweann._update_best_child(child)

        actual = [self.test_tweann.fittest_child, self.test_tweann.fittest_child_ever,
                  self.test_tweann.best_fitness_score, self.test_tweann.best_fitness_score_ever]

        expected = ['Child5', 'Child0', 10, 10]
        self.assertEqual(expected, actual)

    def test_update_children_fitness(self):
        self.test_tweannClassUsage.create_population()
        self.test_tweannClassUsage._update_children_fitness()
        actual = []
        for num_child in range(0, self.test_tweannClassUsage.population_size + 2):
            if self.test_tweannClassUsage._read_data('Child' + str(num_child) + '/Fitness'):
                actual.append(True)
            else:
                actual.append(False)
        expected = [True] * (self.test_tweannClassUsage.population_size + 2)
        self.assertEqual(expected, actual)

    def test_roulette_wheel_selection(self):
        self.test_tweann.create_population()
        for num_child in range(0, self.test_tweann.population_size + 2):
            fitness = random()
            self.test_tweann.total_fitness_score += fitness
            self.test_tweann._write_data('Child' + str(num_child) + '/Fitness', fitness)

        for _ in range(0, 1000):
            self.test_tweann._roulette_wheel_selection()

        self.assertEqual(True, True)

    def test_pick(self):
        self.test_tweann.create_population()
        key = 'Child0/HiddenLayers'
        actual = self.test_tweann._pick(key)
        expected = 0
        self.assertEqual(expected, actual)

    def test_crossover_node_pick(self):
        self.test_tweann.create_child_dataset(self.test_child0)
        self.test_tweann.create_child_dataset(self.test_child3)
        mum = 'Child0'
        dad = 'Child1'
        layer, node, node_key = self.test_tweann._crossover_node_pick(mum, dad)
        if isinstance(layer, int) and isinstance(node, int) and isinstance(node_key, str):
            self.assertEqual(True, True)
        else:
            self.assertEqual(False, True)

    def test_clone(self):
        self.test_tweann.create_population()
        parent = 'Child0'
        self.test_tweann._clone(parent, 5)
        actual_dict = self.test_tweann._get_node_data('Child0/Layer-1/Node0')
        expected_dict = self.test_tweann._get_node_data('Child5/Layer-1/Node0', hdf5_file='Babies.hdf5')
        actual = self.test_tweann._calculate_node_output(actual_dict)
        expected = self.test_tweann._calculate_node_output(expected_dict)

        self.assertAlmostEqual(actual, expected)

        # Todo not used just an idea to try

    def test_inherit(self):
        self.test_tweann.create_child_dataset(self.test_child0)
        self.test_tweann.create_child_dataset(self.test_child3)
        recipient = 'Child0'
        donor = 'Child1'
        node_key = 'Layer0/Node0'
        self.test_tweann._inherit(recipient, donor, node_key, 5)
        actual = self.test_tweann._get_choice_list(['Child0', '', ''])
        expected = self.test_tweann._get_choice_list(['Child5', '', ''], hdf5_file='Babies.hdf5')

        self.assertEqual(True, True)

    def test_add_new_node(self):
        self.test_tweann.create_child_dataset(self.test_child0)
        key = 'Child0/Layer0/Node0'
        num_inputs = 5
        self.test_tweann._add_new_node(key, num_inputs, hdf5_file='Children.hdf5')
        actual = list(self.test_tweann._get_node_data(key).keys())
        expected = self.test_tweann.neuron_chromo

        self.assertEqual(expected, actual)

    def test_check_node_exist_node_exist(self):
        test_child = self.test_child2
        self.test_tweann.create_child_dataset(test_child)
        parent = 'Child0'
        layer = 0
        node = 0
        actual = self.test_tweann._check_node_exist(parent, layer, node)
        expected = True

        self.assertEqual(actual, expected)

    def test_check_node_exist_layer_not_exist(self):
        test_child = self.test_child2
        self.test_tweann.create_child_dataset(test_child)
        parent = 'Child0'
        layer = 3
        node = 0
        actual = self.test_tweann._check_node_exist(parent, layer, node)
        expected = False

        self.assertEqual(actual, expected)

    def test_check_node_exist_layer_exist_node_doesnt(self):
        test_child = self.test_child2
        self.test_tweann.create_child_dataset(test_child)
        parent = 'Child0'
        layer = -1
        node = 5
        actual = self.test_tweann._check_node_exist(parent, layer, node)
        expected = False

        self.assertEqual(expected, actual)

    def test_swap_array(self):
        mum_array = np.array([[0], [1], [2], [3], [4]])
        dad_array = np.array([[10], [11], [21], [31], [41]])
        max_swap = 4
        swap_point = 2
        actual1, actual2 = self.test_tweann._swap_array(mum_array, dad_array, max_swap, swap_point)
        actual = [actual1.tolist(), actual2.tolist()]
        expected = [[[0], [1], [21], [31], [4]], [[10], [11], [2], [3], [41]]]

        self.assertEqual(expected, actual)

    def test_get_max_swap_point_mum_equal_dad(self):
        mum_array = np.array([[0], [1], [2], [3], [4]])
        dad_array = np.array([[10], [11], [21], [31], [41]])

        actual = self.test_tweann._get_max_swap_point(mum_array, dad_array)
        expected = 5

        self.assertEqual(expected, actual)

    def test_get_max_swap_point_mum_greater_dad(self):
        mum_array = np.array([[0], [1], [2], [3], [4]])
        dad_array = np.array([[10], [11], [21]])

        actual = self.test_tweann._get_max_swap_point(mum_array, dad_array)
        expected = 3

        self.assertEqual(expected, actual)

    def test_get_max_swap_point_mum_lesser_dad(self):
        mum_array = np.array([[0], [1], [2]])
        dad_array = np.array([[10], [11], [21], [31], [41]])

        actual = self.test_tweann._get_max_swap_point(mum_array, dad_array)
        expected = 3

        self.assertEqual(expected, actual)

    def test_cross_over_random_swap_point(self):
        mum_array = np.array([[0], [1], [2], [3], [4]])
        dad_array = np.array([[10], [11], [21], [31], [41]])

        mum_array_list = mum_array.tolist()
        dad_array_list = dad_array.tolist()

        actual_baby1, actual_baby2 = self.test_tweann._cross_over_random_swap_point(mum_array, dad_array)

        actual_baby1_list = actual_baby1.tolist()
        actual_baby2_list = actual_baby2.tolist()
        if (mum_array_list != actual_baby1_list) and (dad_array_list != actual_baby2_list) and (
                actual_baby1_list != actual_baby2_list):
            self.assertEqual(True, True)

    def test_get_crossover_node_dict(self):
        self.test_tweann.create_child_dataset(self.test_child0)
        self.test_tweann.create_child_dataset(self.test_child3)
        mum = 'Child0'
        dad = 'Child1'
        node_key = 'Layer-1/Node0'
        mum_dict, dad_dict = self.test_tweann._get_crossover_node_dict(mum, dad, node_key)
        actual = [self.test_tweann._calculate_node_output(mum_dict),
                  self.test_tweann._calculate_node_output(dad_dict)]

        expected_mum_dict = self.test_tweann._get_node_data('Child0/Layer-1/Node0')
        expected_dad_dict = self.test_tweann._get_node_data('Child1/Layer-1/Node0')
        expected = [self.test_tweann._calculate_node_output(expected_mum_dict),
                    self.test_tweann._calculate_node_output(expected_dad_dict)]

        self.assertEqual(expected, actual)

    def test_crossover(self):
        self.assertEqual(False, True)

    def test_crossover_node_element(self):
        mum_dict = {'Weights': np.array([[1.0], [1.0]]),
                    'Bias': np.array([[1]]),
                    'Enabled': 1,
                    'Recurrent': 1,
                    'RecurrentWeight': np.array([[1]]),
                    'Activation': 'linear',
                    'Inputs': np.array([[1], [1]]),
                    'Output': np.array([[1]])
                    }

        dad_dict = {'Weights': np.array([[2], [2]]),
                    'Bias': np.array([[2]]),
                    'Enabled': 0,
                    'Recurrent': 0,
                    'RecurrentWeight': np.array([[2]]),
                    'Activation': 'sigmoid',
                    'Inputs': np.array([[2], [2]]),
                    'Output': np.array([[2]])
                    }
        actual_baby1, actual_baby2 = self.test_tweann._crossover_node_element(mum_dict, dad_dict)
        self.assertEqual(True, True)

    def test_cross_over_swap(self):
        mum_element = 100
        dad_element = 10
        actual_baby1, actual_baby2 = self.test_tweann._cross_over_swap(mum_element, dad_element, rate=0.0)
        actual = [actual_baby1, actual_baby2]
        expected = [dad_element, mum_element]

        self.assertEqual(expected, actual)

    def test_mutate(self, child_num):
        self.assertEqual(False, True)
        # _mutate(self, child_num)

    def test_mutate_node(self):
        self.assertEqual(False, True)

    def test_mutate_node_pick(self):
        self.assertEqual(False, True)

    def test_mutate_node_element(self):
        self.assertEqual(False, True)

    def test_mutate_array(self):
        self.assertEqual(False, True)

    def test_mutate_by_adding_node(self):
        self.assertEqual(False, True)

    def test_increase_child_layer(self):
        self.assertEqual(False, True)

    def test_increase_child_layer_node(self):
        self.assertEqual(False, True)

    def test_new_node_dict(self):
        self.assertEqual(False, True)

    def test_create_new_population(self):
        self.test_tweann.create_population()
        self.test_tweann.fittest_child = 'Child0'
        self.test_tweann.fittest_child_ever = 'Child1'
        self.test_tweann._create_new_population()
        actual = []
        if path.exists('Children.hdf5'):
            actual.append(True)
        else:
            actual.append(True)

        if path.exists('Babies.hdf5'):
            actual.append(False)
        else:
            actual.append(True)

        expected = [True, True]

        self.assertEqual(expected, actual)

    def test_add_best_children(self):
        self.test_tweann.create_population()
        self.test_tweann.fittest_child = 'Child0'
        self.test_tweann.fittest_child_ever = 'Child1'

        self.test_tweann._add_best_children()
        actual = [list(self.test_tweann._get_node_data('Child0/Layer-1/Node0').keys()),
                  list(self.test_tweann._get_node_data('Child1/Layer-1/Node0').keys())]
        expected = [list(self.test_tweann._get_node_data('Child6/Layer-1/Node0', hdf5_file='Babies.hdf5').keys()),
                    list(self.test_tweann._get_node_data('Child7/Layer-1/Node0', hdf5_file='Babies.hdf5').keys())]

        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
