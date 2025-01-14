###################################################
# This file represents the demo for my Recursive
# Descent (RD) algorithm
#
# This algorithm uses ideas from the following paper:
# https://www.researchgate.net/publication/325075500_Rumor_Detection_on_Twitter_with_Tree-structured_Recursive_Neural_Networks
#
# and resources from their codebase available here:
# https://github.com/majingCUHK/Rumor_RvNN
# 
# which is under the MIT license.
#
# Author: Andrew Fisher
###################################################

import sys
import time
import datetime
import numpy as np

import RDescent

# Classifications as defined in the original codebase
labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']

def main(dataset):
    print("Start of the Recursive Descent (RD) algorithm demo for Twitter rumor detection!\n")
    
    # Define dataset paths
    treePath = "../resource/data.TD_RvNN.vol_5000.txt"
    labelPath = "../resource/" + dataset + "_label_All.txt"
    trainPath = "../nfold/RNNtrainSet_" + dataset + "3_tree.txt"
    testPath = "../nfold/RNNtestSet_" + dataset + "3_tree.txt"

    # Define variables
    epochs = 600
    lr = 0.005

    # Load the data
    print("Loading the datasets... ")
    start_time = time.time()
    
    treeDictionary = loadTreeData(treePath)
    labelDictionary = loadLabelData(labelPath)
    tree_train, word_train, index_train, parent_num_train, y_train = loadTrainData(treeDictionary, labelDictionary, trainPath)
    tree_test, word_test, index_test, parent_num_test, y_test = loadTestData(treeDictionary, labelDictionary, testPath)
    print("Completed in %s seconds!\n" % (time.time() - start_time))

    # Initialize the algorithm
    print("Initializing the algorithm... ")
    start_time = time.time()

    model = RDescent.algorithm(5000)
    print("Completed in %s seconds!\n" % (time.time() - start_time))

    # Run the epochs
    losses_5, losses = [], []
    num_examples_seen = 0
    for epoch in range(epochs):
        indexs = [i for i in range(len(y_train) - 1)]
        for i in indexs:
            loss, pred_y = model.train_step_up(word_train[i], index_train[i], parent_num_train[i], tree_train[i], y_train[i], lr, True) #Top down
            losses.append(np.round(loss,2))

            j = i + 1
            loss, pred_y = model.train_step_up(word_train[j], index_train[j], parent_num_train[j], tree_train[j], y_train[j], lr, False) #Bottom up
            losses.append(np.round(loss,2))

            num_examples_seen += 1
        print("epoch=" + str(epoch) + ": loss="  + str(np.mean(losses)))
        sys.stdout.flush()

        # Every five epochs, test the algorithm and see if the learning rate should be adjusted
        if epoch % 5 == 0:
            losses_5.append((num_examples_seen, np.mean(losses))) 
            cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(cur_time + ": Loss after num_examples_seen=" + str(num_examples_seen) + " epoch=: " + str(epoch) + " " + str(np.mean(losses)))

            sys.stdout.flush()
            prediction = []
            for j in range(len(y_test) - 1):
                prediction.append(model.predict_up(word_test[j], index_test[j], parent_num_test[j], tree_test[j], True)) #Top down
                k = j + 1
                prediction.append(model.predict_up(word_test[k], index_test[k], parent_num_test[k], tree_test[k], False)) #Bottom up

            res = evaluation_4class(prediction, y_test) 
            print('results:' + " " + prediction[y_test])
            sys.stdout.flush()

            if len(losses_5) > 1 and losses_5[-1][1] > losses_5[-2][1]:
                lr = lr * 0.5   
                print("Setting learning rate to " + str(lr))
                sys.stdout.flush()
        losses = []

# Using code from the original codebase's "Main_TD_RvNN.py" file in the 'loadData' method
def loadTreeData(treePath):
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        parent_num, maxL = int(line.split('\t')[3]), int(line.split('\t')[4])  
        vec =  line.split('\t')[5] 
        if eid not in treeDic:
           treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent':indexP, 'parent_num':parent_num, 'maxL':maxL, 'vec':vec}
    return treeDic

# Using code from the original codebase's "Main_TD_RvNN.py" file in the 'loadData' method
def loadLabelData(labelPath):
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        label, eid = line.split('\t')[0], line.split('\t')[2]
        labelDic[eid] = label.lower()
    return labelDic

# Using code from the original codebase's "Main_TD_RvNN.py" file in the 'loadData' method
def loadTrainData(treeDic, labelDic, trainPath):
    tree_train, word_train, index_train, y_train, parent_num_train, c = [], [], [], [], [], 0

    for eid in open(trainPath):
        eid = eid.rstrip()
        
        if eid not in labelDic: continue
        if eid not in treeDic: continue 
        if len(treeDic[eid]) <= 0: 
           continue

        label = labelDic[eid]
        y = loadLabel(label)
        y_train.append(y)

        x_word_td, x_index_td, tree_td, x_word_bu, x_index_bu, tree_bu, parent_num = constructTree(treeDic[eid])

        tree_train.append(tree)
        tree_train.append(tree)

        word_train.append(x_word_td)
        word_train.append(x_word_bu)

        index_train.append(x_index_td)
        index_train.append(x_index_bu)

        parent_num_train.append(parent_num)
        parent_num_train.append(parent_num)

        c += 1
    return tree_train, word_train, index_train, parent_num_train, y_train

# Using code from the original codebase's "Main_TD_RvNN.py" file in the 'loadData' method
def loadTestData(treeDic, labelDic, testPath):
    tree_test, word_test, index_test, parent_num_test, y_test, c = [], [], [], [], [], 0
    for eid in open(testPath):
        eid = eid.rstrip()

        if eid not in labelDic: continue
        if eid not in treeDic: continue 
        if len(treeDic[eid]) <= 0: 
           continue        
      
        label = labelDic[eid]
        y = loadLabel(label)
        y_test.append(y)

        x_word_td, x_index_td, tree_td, x_word_bu, x_index_bu, tree_bu, parent_num  = constructTree(treeDic[eid])

        tree_test.append(tree)
        tree_test.append(tree)

        word_test.append(x_word_td)
        word_test.append(x_word_bu)

        index_test.append(x_index_td)
        index_test.append(x_index_bu)

        parent_num_test.append(parent_num)
        parent_num_test.append(parent_num)

        c += 1
    return tree_test, word_test, index_test, parent_num_test, y_test

# Using code from the original codebase's "Main_TD_RvNN.py" file
def loadLabel(label):
    if label in labelset_nonR:
       return [1,0,0,0]
    if label in labelset_f:
       return [0,1,0,0] 
    if label in labelset_t:
       return [0,0,1,0] 
    if label in labelset_u:
       return [0,0,0,1] 
    return None

# Using code from the original codebase's "Main_TD_RvNN.py" file
def constructTree(tree):
    index2node = {}
    for i in tree:
        node = RDescent.Node(i)
        index2node[i] = node

    for j in tree:
        indexC = j 
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix( tree[j]['vec'], tree[j]['maxL'] )

        nodeC.index = wordIndex
        nodeC.word = wordFreq

        if not indexP == 'None':
           nodeP = index2node[int(indexP)]
           nodeC.parent = nodeP
           nodeP.children.append(nodeC)
        else:
           root = nodeC

    parent_num = tree[j]['parent_num'] 
    ini_x, ini_index = str2matrix( "0:0", tree[j]['maxL'] )

    x_word_td, x_index_td, tree_td = RDescent.gen_nn_inputs_td(root, ini_x) 
    x_word_bu, x_index_bu, tree_bu = RDescent.gen_nn_inputs_bu(root, max_degree=parent_num, only_leaves_have_vals=False)

    return x_word_td, x_index_td, tree_td, x_word_bu, x_index_bu, tree_bu, parent_num  

# Using code from the original codebase's "Main_TD_RvNN.py" file
def str2matrix(Str, MaxL):
    wordFreq, wordIndex = [], []
    l = 0
    for pair in Str.split(' '):
        wordFreq.append(float(pair.split(':')[1]))
        wordIndex.append(int(pair.split(':')[0]))
        l += 1
    ladd = [ 0 for i in range( MaxL-l ) ]
    wordFreq += ladd 
    wordIndex += ladd 
    return wordFreq, wordIndex 

if __name__ == '__main__':
    dataset = sys.argv[1]

    if dataset == None or (dataset != "Twitter15" and dataset != "Twitter16"):
        print("Please indicate whether you'd like to use Twitter15 or Twitter16 as a command line argument")
    else:
        main(dataset)