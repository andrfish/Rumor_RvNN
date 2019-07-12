###################################################
# This file represents my Recursive Descent (RD)
# algorithm for Twitter rumor detection
#
# Author: Andrew Fisher
###################################################

#!/usr/bin/env python
import numpy as np
import tensorflow as tf

def gen_nn_inputs(root_node, ini_word):
    tree = [[0, root_node.tree]] 

    X_word, X_index = [root_node.word], [root_node.length]

    internal_tree, internal_word, internal_index  = _get_tree_path(root_node)

    tree.extend(internal_tree)    
    X_word.extend(internal_word)
    X_index.extend(internal_index)
    X_word.append(ini_word)
 
    return (np.array(X_word, dtype='float32'),
            np.array(X_index, dtype='int32'),
            np.array(tree, dtype='int32'))     

def _get_tree_path(root_node):
    if not root_node.children:
        return [], [], []

    layers = []
    layer = [root_node]
    while layer:
        layers.append(layer[:])
        next_layer = []
        [next_layer.extend([child for child in node.children if child])
         for node in layer]
        layer = next_layer

    tree = []
    word = []
    index = []
    for layer in layers:
        for node in layer:
            if not node.children:
               continue 

            for child in node.children:
                tree.append([node.tree, child.tree])
                word.append(child.word if child.word is not None else -1)
                index.append(child.length if child.length is not None else -1)

    return tree, word, index

# Define a node object for the tweet tree
class Node(object):
    def __init__(self, tree=None):
        self.children = list()
        self.parent = None

        self.word = list()
        self.length = -1    #index

        self.tree = tree    #idx

# Define a tree object for the algorithm
class algorithm(object):
    _sess = tf.Session()

    def __init__(self, word_dim):
        self.word_dim = word_dim

        # Using defaults from original codebase's "TD_RvNN.py" file
        self.hidden_dim = 5
        self.Nclass = 4
        self.degree = 2
        self.momentum = 0.9
        self.irregular_tree = True
        self.params = []

        # Initialize algorithm variables in Tensorflow
        self.word_freq = tf.Variable([[]], tf.int32)    #x_word
        self.word_index = tf.Variable([[]], tf.int32)   #x_index
        self.tree = tf.Variable([[]], tf.float64)
        self.out_shape = tf.Variable([], tf.int32)      #y

        self.num_parent = tf.Variable(-1, tf.int32)
        self.num_nodes = tf.shape(self.word_freq)
        self.num_child = self.num_nodes - self.num_parent - 1

        # Initialize Tensorflow functions
        self.tree_states = self.compute_tree(self.word_freq, self.word_index, self.num_parent, self.tree)
        self.final_state = tf.reduce_max(self.tree_states, axis=0)
        self.output = self.create_output_fn()
        self.pred_y = self.output_fn(self.final_state)
        self.loss = self.loss_fn(self.y, self.pred_y)

        self.learning_rate = tf.Variable(-1, tf.float64)

        # Initialize Tensorflow functions from Theano
        self.tree_states_test = self.compute_tree_test(self.word_freq, self.x_index, self.tree)
        train_inputs = tf.Variable([self.word_freq, self.word_index, self.num_parent, self.tree, self.out_shape, self.learning_rate], tf.Tensor)
        updates = self.gradient_descent(self.loss)

        self._train = TensorFlowTheanoFunction(train_inputs, [self.loss, self.pred_y], updates=updates)

        self._evaluate = TensorFlowTheanoFunction([self.word_freq, self.word_index, self.num_parent, self.tree], self.final_state)
        self._evaluate2 = TensorFlowTheanoFunction([self.word_freq, self.word_index, self.num_parent, self.tree], self.tree_states)
        self._evaluate3 = TensorFlowTheanoFunction([self.word_freq, self.word_index, self.num_parent, self.tree], self.tree_states_test)

        self._predict = TensorFlowTheanoFunction([self.word_freq, self.word_index, self.num_parent, self.tree], self.pred_y)

    # Using code from original codebase's "TD_RvNN.py" file
    def train_set_up(self, word_freq, word_index, num_parent, tree, output_shape, learning_rate):
        return self._train(word_freq, word_index, num_parent, tree, output_shape, learning_rate)
    
    # Using code from original codebase's "TD_RvNN.py" file
    def evaluate(self, word_freq, word_index, num_parent, tree):
        return self._evaluate(word_freq, word_index, num_parent, tree)

    # Using code from original codebase's "TD_RvNN.py" file
    def predict_up(self, word_freq, word_index, num_parent, tree):
        return self._predict(word_freq, word_index, num_parent, tree)

    # Using code from original codebase's "TD_RvNN.py" file
    def init_matrix(self, shape):
        return np.random.normal(scale=0.1, size=shape).astype(np.float)

    # Using code from original codebase's "TD_RvNN.py" file
    def init_vector(self, shape):
        return np.zeros(shape, dtype=np.float)

    # Using code from original codebase's "TD_RvNN.py" file
    def create_output_fn(self):
        self.W_out = self.init_matrix([self.Nclass, self.hidden_dim])
        self.b_out = self.init_vector([self.Nclass])
        self.params.extend([self.W_out, self.b_out])

        def fn(final_state):
            return tf.nn.softmax(self.W_out.dot(final_state) + self.b_out)

        return fn

    # Using code from original codebase's "TD_RvNN.py" file
    def create_recursive_unit(self):
        self.E = self.init_matrix([self.hidden_dim, self.word_dim])
        self.W_z = self.init_matrix([self.hidden_dim, self.hidden_dim])
        self.U_z = self.init_matrix([self.hidden_dim, self.hidden_dim])
        self.b_z = self.init_vector([self.hidden_dim])
        self.W_r = self.init_matrix([self.hidden_dim, self.hidden_dim])
        self.U_r = self.init_matrix([self.hidden_dim, self.hidden_dim])
        self.b_r = self.init_vector([self.hidden_dim])
        self.W_h = self.init_matrix([self.hidden_dim, self.hidden_dim])
        self.U_h = self.init_matrix([self.hidden_dim, self.hidden_dim])
        self.b_h = self.init_vector([self.hidden_dim])
        self.params.extend([self.E, self.W_z, self.U_z, self.b_z, self.W_r, self.U_r, self.b_r, self.W_h, self.U_h, self.b_h])
        def unit(word, index, parent_h):
            child_xe = self.E[:,index].dot(word)
            z = tf.nn.sigmoid(self.W_z.dot(child_xe)+self.U_z.dot(parent_h)+self.b_z)
            r = T.nnet.sigmoid(self.W_r.dot(child_xe)+self.U_r.dot(parent_h)+self.b_r)
            c = T.tanh(self.W_h.dot(child_xe)+self.U_h.dot(parent_h * r)+self.b_h)
            h = z*parent_h + (1-z)*c
            return h
        return unit

    def compute_tree(self, x_word, x_index, num_parent, tree):
        self.recursive_unit = self.create_recursive_unit()

        def ini_unit(x):
            return self.init_vector([self.hidden_dim])

        init_node_h, _ = tf.scan(
            fn=ini_unit,
            elems=[ x_word ])

        # use recurrence to compute internal node hidden states
        def _recurrence(x_word, x_index, node_info, node_h, last_h):
            parent_h = node_h[node_info[0]]
            child_h = self.recursive_unit(x_word, x_index, parent_h)

            node_h = tf.concat([node_h[:node_info[1]],
                                    child_h.reshape([1, self.hidden_dim]),
                                    node_h[node_info[1]+1:] ])
            return node_h, child_h

        dummy = tself.init_vector([self.hidden_dim])
        (_, child_hs), _ = tf.scan(
            fn=_recurrence,
            initializer=[init_node_h, dummy],
            elems=[x_word[:-1], x_index, tree])

        return child_hs[num_parent-1:]

    def compute_tree_test(self, x_word, x_index, tree):
        self.recursive_unit = self.create_recursive_unit()
        def ini_unit(x):
            return self.init_vector([self.hidden_dim])
        init_node_h, _ = tf.scan(
            fn=ini_unit,
            elems=[ x_word ])

        def _recurrence(x_word, x_index, node_info, node_h, last_h):
            parent_h = node_h[node_info[0]]
            child_h = self.recursive_unit(x_word, x_index, parent_h)

            node_h = T.concatenate([node_h[:node_info[1]],
                                    child_h.reshape([1, self.hidden_dim]),
                                    node_h[node_info[1]+1:] ])
            return node_h, child_h

        dummy = self.init_vector([self.hidden_dim])
        (_, child_hs), _ = tf.scan(
            fn=_recurrence,
            initializer=[init_node_h, dummy],
            elems=[x_word[:-1], x_index, tree])
        return child_hs
        
    def loss_fn(self, y, pred_y):
        return tf.math.reduce_sum(T.sqr(y - pred_y))

    def gradient_descent(self, loss):
        grad = tf.gradients(loss, self.params)
        self.momentum_velocity_ = [0.] * len(grad)
        grad_norm = tf.math.sqrt(sum(map(lambda x: tf.math.sqrt(x).sum(), grad)))
        updates = OrderedDict()
        not_finite = tf.math.logical_or(tf.math.is_nan(grad_norm), T.math.is_inf(grad_norm))
        scaling_den = T.math.maximum(5.0, grad_norm)
        for n, (param, grad) in enumerate(zip(self.params, grad)):
            grad = tf.cond(not_finite, 0.1 * param,
                            grad * (5.0 / scaling_den))
            velocity = self.momentum_velocity_[n]
            update_step = self.momentum * velocity - self.learning_rate * grad
            self.momentum_velocity_[n] = update_step
            updates[param] = param + update_step
        return updates

# Taken from the following stackoverflow answer:
# https://stackoverflow.com/a/40430597
class TensorFlowTheanoFunction(object):   
  def __init__(self, inputs, outputs, updates=()):
    self._inputs = inputs
    self._outputs = outputs
    self._updates = updates

  def __call__(self, *args, **kwargs):
    feeds = {}
    for (argpos, arg) in enumerate(args):
      feeds[self._inputs[argpos]] = arg
    try:
      outputs_identity = [tf.identity(output) for output in self._outputs]
      output_is_list = True
    except TypeError:
      outputs_identity = [tf.identity(self._outputs)]
      output_is_list = False
    with tf.control_dependencies(outputs_identity):
      assign_ops = [tf.assign(variable, replacement) 
                    for variable, replacement in self._updates]
    outputs_list = tf.get_default_session().run(
        outputs_identity + assign_ops, feeds)[:len(outputs_identity)]
    if output_is_list:
      return outputs_list
    else:
      assert len(outputs_list) == 1
      return outputs_list[0]