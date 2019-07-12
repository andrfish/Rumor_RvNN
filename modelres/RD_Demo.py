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

        x_word, x_index, tree, parent_num = constructTree(treeDic[eid])
        tree_train.append(tree)
        word_train.append(x_word)
        index_train.append(x_index)
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

        x_word, x_index, tree, parent_num = constructTree(treeDic[eid])
        tree_test.append(tree)
        word_test.append(x_word)  
        index_test.append(x_index) 
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
    x_word, x_index, tree = RDescent.gen_nn_inputs(root, ini_x) 
    return x_word, x_index, tree, parent_num  

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