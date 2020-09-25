import unittest
from LearnAfter1MTries.TWEANN_v2.Tweann import *
from os import path


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
                            'Child0/Fitness': 0,
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
                            'Child0/Fitness': 0
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
                            'Child0/Layer1/Node0/Weights': np.random.rand(self.num_inputs, 1),
                            'Child0/Layer1/Node0/Bias': np.random.rand(1, 1),
                            'Child0/Layer1/Node0/Enabled': 1,
                            'Child0/Layer1/Node0/Recurrent': choice([0, 1]),
                            'Child0/Layer1/Node0/RecurrentWeight': np.random.rand(1, 1),
                            'Child0/Layer1/Node0/Activation': choice(self.activationFunc),
                            'Child0/Layer1/Node0/Inputs': np.array([[2.0], [2.0]]),
                            'Child0/Layer1/Node0/Output': np.zeros((1, 1)),
                            'Child0/Layer1/Node1/Weights': np.random.rand(self.num_inputs, 1),
                            'Child0/Layer1/Node1/Bias': np.random.rand(1, 1),
                            'Child0/Layer1/Node1/Enabled': 1,
                            'Child0/Layer1/Node1/Recurrent': choice([0, 1]),
                            'Child0/Layer1/Node1/RecurrentWeight': np.random.rand(1, 1),
                            'Child0/Layer1/Node1/Activation': choice(self.activationFunc),
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
                            'Child0/Fitness': 0
                            }

    def tearDown(self):
        if path.exists('Children.hdf5'):
            remove('Children.hdf5')

    def test_create_init_child_chromo_keys(self):
        actual = self.test_tweann.create_init_child_chromo('Child0').keys()
        expected = self.test_child0.keys()
        self.assertEqual(expected, actual)

    def test_create_init_child_chromo_dataTypes(self):
        child = self.test_tweann_2.create_init_child_chromo('Child0')
        actual = []
        for item in child:
            if isinstance(child[item], str):
                actual.append(True)
            if isinstance(child[item], np.ndarray):
                actual.append(child[item].shape)
            if isinstance(child[item], int):
                actual.append(True)
            if isinstance(child[item], type(None)):
                actual.append(None)

        expected = [(self.num_inputs, 1), (1, 1), True, True, (self.num_outputs, 1), True, (2, 1),
                    (self.num_outputs, 1),
                    (self.num_inputs, 1), (1, 1), True, True, (self.num_outputs, 1), True, (2, 1),
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

        for index in range(0, 5):
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
        print(actual)
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


if __name__ == '__main__':
    unittest.main()
