###################################################
# This file represents my Recursive Descent (RD)
# algorithm for Twitter rumor detection
#
# This code was modified from the original
# codebase's "TD_RvNN.py" file
#
# Author: Andrew Fisher
###################################################

#!/usr/bin/env python
import numpy as np
import theano
from theano import tensor as T
from collections import OrderedDict
from theano.tensor.signal.pool import pool_2d

# This method creates a input for the neural network, based on the passed root node
def gen_nn_inputs_td(root_node, ini_word):
    tree = [[0, root_node.tree]] 

    X_word, X_index = [root_node.word], [root_node.index]

    internal_tree, internal_word, internal_index  = _get_tree_path(root_node)

    tree.extend(internal_tree)    
    X_word.extend(internal_word)
    X_index.extend(internal_index)
    X_word.append(ini_word)
  
    return (np.array(X_word, dtype='float32'),
            np.array(X_index, dtype='int32'),
            np.array(tree, dtype='int32'))

def gen_nn_inputs_bu(root_node, max_degree=None, only_leaves_have_vals=True, with_labels=False):
    _clear_indices(root_node)

    X_word, X_index = _get_leaf_vals(root_node)
    tree, internal_word, internal_index = _get_tree_traversal(root_node, len(X_word), max_degree)

    X_word.extend(internal_word)
    X_index.extend(internal_index)
    if max_degree is not None:
        assert all(len(t) == max_degree + 1 for t in tree)
 
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
                index.append(child.index if child.index is not None else -1)

    return tree, word, index     

def _clear_indices(root_node):
    root_node.idx = None
    [_clear_indices(child) for child in root_node.children if child]


def _get_leaf_vals(root_node):
    all_leaves = []
    layer = [root_node]
    while layer:
        next_layer = []
        for node in layer:
            if not node.children:
                all_leaves.append(node)
            else:
                next_layer.extend([child for child in node.children[::-1] if child])
        layer = next_layer

    X_word = []
    X_index = []
    for idx, leaf in enumerate(reversed(all_leaves)):
        leaf.idx = idx
        X_word.append(leaf.word)
        X_index.append(leaf.index)
    return X_word, X_index

def _get_tree_traversal(root_node, start_idx=0, max_degree=None):
    
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
    internal_word = []
    internal_index = []
    idx = start_idx
    for layer in reversed(layers):
        for node in layer:
            if node.idx is not None:
                # must be leaf
                assert all(child is None for child in node.children)
                continue

            child_idxs = [(child.idx if child else -1)
                          for child in node.children]  ## idx of child node
            if max_degree is not None:
                child_idxs.extend([-1] * (max_degree - len(child_idxs)))
            assert not any(idx is None for idx in child_idxs)

            node.idx = idx
            tree.append(child_idxs + [node.idx])
            internal_word.append(node.word if node.word is not None else -1)
            internal_index.append(node.index if node.index is not None else -1)
            idx += 1

    return tree, internal_word, internal_index

# Used to store information for each node in the tree
class Node(object):
    def __init__(self, idx=None):
        self.children = []
        self.tree = idx
        self.word = []
        self.index = []
        self.parent = None

# Used for the Recursive Descent (RD) algorithm
class algorithm(object):

    # Initializes the algorithm
    def __init__(self, word_dim, hidden_dim=5, Nclass=4,
                degree=2, momentum=0.9,
                 trainable_embeddings=True,
                 labels_on_nonroot_nodes=False,
                 irregular_tree=True):   
        assert word_dim > 1 and hidden_dim > 1

        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.Nclass = Nclass
        self.degree = degree
        self.momentum = momentum
        self.irregular_tree = irregular_tree

        self.params_td = []
        self.params_bu = []

        self.word_freq = T.matrix(name='word_freq')
        self.word_idx = T.imatrix(name='word_idx')
        self.tree = T.imatrix(name='tree')
        self.y = T.ivector(name='y')
        self.num_parent = T.iscalar(name='num_parent')

        self.num_nodes_td = T.shape(self.word_freq)
        self.num_nodes_bu = self.word_freq.shape[0]

        self.num_child = self.num_nodes_td - self.num_parent-1

        self.tree_states_td = self.compute_tree_td(self.word_freq, self.word_idx, self.num_parent, self.tree)
        self.tree_states_bu = self.compute_tree_bu(self.word_freq, self.word_idx, self.tree)

        self.final_state_td = self.tree_states_td.max(axis=0)
        self.final_state_bu = self.tree_states_bu[-1]

        self.output_fn_td = self.create_output_fn_td()
        self.output_fn_bu = self.create_output_fn_bu()
        
        self.pred_y_td = self.output_fn_td(self.final_state_td)
        self.pred_y_bu = self.output_fn_bu(self.final_state_bu)

        self.loss_td = self.loss_fn(self.y, self.pred_y_td)
        self.loss_bu = self.loss_fn(self.y, self.pred_y_bu)

        self.learning_rate = T.scalar('learning_rate')

        train_inputs_td = [self.word_freq, self.word_idx, self.num_parent, self.tree, self.y, self.learning_rate]
        train_inputs_bu = [self.word_freq, self.word_idx, self.tree, self.y, self.learning_rate]

        updates_td = self.gradient_descent(self.loss_td, True)
        updates_bu = self.gradient_descent(self.loss_bu, False)

        self._train_td = theano.function(train_inputs_td,
                                      [self.loss_td, self.pred_y_td],
                                      updates=updates_td)
        self._train_bu = theano.function(train_inputs_bu,
                                      [self.loss_bu, self.pred_y_bu],
                                      updates=updates_bu)

        self._evaluate_td = theano.function([self.word_freq, self.word_idx, self.num_parent, self.tree], self.final_state_td)
        self._evaluate_bu = theano.function([self.word_freq, self.word_idx, self.tree], self.final_state_bu)

        self._predict_td = theano.function([self.word_freq, self.word_idx, self.num_parent, self.tree], self.pred_y_td)
        self._predict_bu = theano.function([self.word_freq, self.word_idx, self.tree], self.pred_y_bu)
    
    # This method steps through an epoch
    def train_step_up(self, x_word, x_index, num_parent, tree, y, lr, td):
        if td:
            loss, pred_y = self._train_td(x_word, x_index, num_parent, tree, y, lr)
        else:
            loss, pred_y = self._train_bu(x_word, x_index, tree[:, :-1], y, lr)

        return loss, pred_y
        
    # This method evaluates the accuracy of the current state of the algorithm
    def evaluate(self,  x_word, x_index, num_parent, tree, td):
        if td:
            evalr = self._evaluate_td(x_word, x_index, num_parent, tree)
        else:
            x, tree = gen_nn_inputs(root_node, max_degree=self.degree, only_leaves_have_vals=False)
            self._check_input(x, tree)
            evalr = self._evaluate_bu(x, tree[:, :-1])

        return evalr

    # This method predicts rumors using the current state of the algorithm
    def predict_up(self, x_word, x_index, num_parent, tree, td):
        if td:
            pred = self._predict_td(x_word, x_index, num_parent, tree)
        else:
            pred = self._predict_bu(x_word, x_index, tree[:, :-1])

        return pred

    # This method initializes an empty matrix of shape 'shape'
    def init_matrix(self, shape):
        return np.random.normal(scale=0.1, size=shape).astype(dtype='float32')

    # This method initializes a vector of value 'shape'
    def init_vector(self, shape):
        return np.zeros(shape, dtype='float32')

    # This method returns a Theano function that will be used to calculate the output
    def create_output_fn_td(self):
        self.W_out_td = theano.shared(self.init_matrix([self.Nclass, self.hidden_dim]))
        self.b_out_td = theano.shared(self.init_vector([self.Nclass]))
        self.params_td.extend([self.W_out_td, self.b_out_td])

        def fn(final_state):
            return T.nnet.softmax( self.W_out_td.dot(final_state)+ self.b_out_td )
        return fn

    # This method returns a Theano function that will be used to calculate the output
    def create_output_fn_bu(self):
        self.W_out_bu = theano.shared(self.init_matrix([self.Nclass, self.hidden_dim]))
        self.b_out_bu = theano.shared(self.init_vector([self.Nclass]))
        self.params_bu.extend([self.W_out_bu, self.b_out_bu])

        def fn(final_state):
            return T.nnet.softmax( self.W_out_bu.dot(final_state)+ self.b_out_bu )
        return fn

    # This method returns a Theano function that recurses through the tree from the parent
    def create_recursive_unit_td(self):
        self.E_td = theano.shared(self.init_matrix([self.hidden_dim, self.word_dim]))
        self.W_z_td = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.U_z_td = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_z_td = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_r_td = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.U_r_td = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_r_td = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_h_td = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.U_h_td = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_h_td = theano.shared(self.init_vector([self.hidden_dim]))
        self.params_td.extend([self.E_td, self.W_z_td, self.U_z_td, self.b_z_td, self.W_r_td, self.U_r_td, self.b_r_td, self.W_h_td, self.U_h_td, self.b_h_td])
        def unit(word, index, parent_h):
            child_xe = self.E_td[:,index].dot(word)
            z = T.nnet.hard_sigmoid(self.W_z_td.dot(child_xe)+self.U_z_td.dot(parent_h)+self.b_z_td)
            r = T.nnet.hard_sigmoid(self.W_r_td.dot(child_xe)+self.U_r_td.dot(parent_h)+self.b_r_td)
            c = T.tanh(self.W_h_td.dot(child_xe)+self.U_h_td.dot(parent_h * r)+self.b_h_td)
            h = z*parent_h + (1-z)*c
            return h
        return unit

    # This method returns a Theano function to compute the state of each children under the parent
    def compute_tree_td(self, x_word, x_index, num_parent, tree):
        self.recursive_unit_td = self.create_recursive_unit_td()
        def ini_unit(x):
            return theano.shared(self.init_vector([self.hidden_dim]))
        
        init_node_h, _ = theano.scan(
            fn=ini_unit,
            sequences=[ x_word ])

        def _recurrence(x_word, x_index, node_info, node_h, last_h):
            parent_h = node_h[node_info[0]]
            child_h = self.recursive_unit_td(x_word, x_index, parent_h)
            node_h = T.concatenate([node_h[:node_info[1]],
                                    child_h.reshape([1, self.hidden_dim]),
                                    node_h[node_info[1]+1:] ])
            return node_h, child_h

        dummy = theano.shared(self.init_vector([self.hidden_dim]))
        (_, child_hs), _ = theano.scan(
            fn=_recurrence,
            outputs_info=[init_node_h, dummy],
            sequences=[x_word[:-1], x_index, tree])

        return child_hs[num_parent-1:]

    def create_recursive_unit_bu(self):
        self.E_bu = theano.shared(self.init_matrix([self.hidden_dim, self.word_dim]))
        self.W_z_bu = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.U_z_bu = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_z_bu = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_r_bu = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.U_r_bu = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_r_bu = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_h_bu = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.U_h_bu = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_h_bu = theano.shared(self.init_vector([self.hidden_dim]))
        self.params_bu.extend([self.E_bu, self.W_z_bu, self.U_z_bu, self.b_z_bu, self.W_r_bu, self.U_r_bu, self.b_r_bu, self.W_h_bu, self.U_h_bu, self.b_h_bu])
        def unit(parent_word, parent_index, child_h, child_exists):
            h_tilde = T.sum(child_h, axis=0)
            parent_xe = self.E_bu[:,parent_index].dot(parent_word)
            z = T.nnet.hard_sigmoid(self.W_z_bu.dot(parent_xe)+self.U_z_bu.dot(h_tilde)+self.b_z_bu)
            r = T.nnet.hard_sigmoid(self.W_r_bu.dot(parent_xe)+self.U_r_bu.dot(h_tilde)+self.b_r_bu)
            c = T.tanh(self.W_h_bu.dot(parent_xe)+self.U_h_bu.dot(h_tilde * r)+self.b_h_bu)
            h = z*h_tilde + (1-z)*c
            return h
        return unit

    def create_leaf_unit(self):
        dummy = 0 * theano.shared(self.init_vector([self.degree, self.hidden_dim]))
        def unit(leaf_word, leaf_index):
            return self.recursive_unit_bu( leaf_word, leaf_index, dummy, dummy.sum(axis=1))
        return unit
    def compute_tree_bu(self, x_word, x_index, tree):
        self.recursive_unit_bu = self.create_recursive_unit_bu()
        self.leaf_unit = self.create_leaf_unit()
        num_parents = tree.shape[0]  # num internal nodes
        num_leaves = x_word.shape[0] - num_parents

        # compute leaf hidden states
        leaf_h, _ = theano.map(
            fn=self.leaf_unit,
            sequences=[ x_word[:num_leaves], x_index[:num_leaves] ])
        if self.irregular_tree:
            init_node_h = T.concatenate([leaf_h, leaf_h, leaf_h], axis=0)
        else:
            init_node_h = leaf_h

        # use recurrence to compute internal node hidden states
        def _recurrence(x_word, x_index, node_info, t, node_h, last_h):
            child_exists = node_info > -1
            offset = 2*num_leaves * int(self.irregular_tree) - child_exists * t ### offset???
            child_h = node_h[node_info + offset] * child_exists.dimshuffle(0, 'x') ### transpose??
            parent_h = self.recursive_unit_bu(x_word, x_index, child_h, child_exists)
            node_h = T.concatenate([node_h,
                                    parent_h.reshape([1, self.hidden_dim])])
            return node_h[1:], parent_h

        dummy = theano.shared(self.init_vector([self.hidden_dim]))
        (_, parent_h), _ = theano.scan(
            fn=_recurrence,
            outputs_info=[init_node_h, dummy],
            sequences=[x_word[num_leaves:], x_index[num_leaves:], tree, T.arange(num_parents)],
            n_steps=num_parents)

        return T.concatenate([leaf_h, parent_h], axis=0)
        
    # This method defines the loss function
    def loss_fn(self, y, pred_y):
        return T.sum(T.sqr(y - pred_y))

    # This method defines the gradient function when updating the algorithm on each epoch
    def gradient_descent(self, loss, td):
        grad = T.grad(loss, self.params_td if td else self.params_bu)
        self.momentum_velocity_ = [0.] * len(grad)
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grad)))
        updates = OrderedDict()
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        scaling_den = T.maximum(5.0, grad_norm)
        for n, (param, grad) in enumerate(zip(self.params_td if td else self.params_bu, grad)):
            grad = T.switch(not_finite, 0.1 * param,
                            grad * (5.0 / scaling_den))
            velocity = self.momentum_velocity_[n]
            update_step = self.momentum * velocity - self.learning_rate * grad
            self.momentum_velocity_[n] = update_step
            updates[param] = param + update_step
        return updates
        