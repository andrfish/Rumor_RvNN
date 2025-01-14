# Fork Description

This repository is a fork of the source code repository as described below. In this fork, we made original code compatible with Python 3.7 and to work on GPU instead of CPU.

# Fork Dependencies
Please ensure that you have the following installed in addition to the other dependencies:
- CUDA 9.0
- cuDNN v7.6.1 (June 24, 2019), for CUDA 9.0
- libgpuarray 0.7.1 (conda install -c mila-udem pygpu)
- mingw (conda install m2w64-toolchain)
- libpython (conda install libpython)

Then, copy the .theanroc config file in the repository to your home folder. Make sure to update the paths in it appropriately.

# Paper of the source codes released:

Jing Ma, Wei Gao, Kam-Fai Wong. Rumor Detection on Twitter with Tree-structured Recursive Neural Networks. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, ACL 2018.

# Datasets:

The datasets used in the experiments were based on the two publicly available Twitter datasets released by Ma et al. (2017):

Jing Ma, Wei Gao, Kam-Fai Wong. Detect Rumors in Microblog Posts Using Propagation Structure via Kernel Learning. ACL 2017.

In the 'resource' folder we provide the pre-processed data files used for our experiments. The raw datasets can be downloaded from https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0. For details about the datasets please contact Jing at: majing at se dot cuhk dot edu dot hk.

The datafile is in a tab-sepreted column format, where each row corresponds to a tweet. Consecutive columns correspond to the following pieces of information:

1: root-id -- an unique identifier describing the tree (tweetid of the root);

2: index-of-parent-tweet -- an index number of the parent tweet for the current tweet;

3: index-of-the-current-tweet -- an index number of the current tweet;

4: parent-number -- the total number of the parent node in the tree that the current tweet is belong to;

5: text-length -- the maximum length of all the texts from the tree that the current tweet is belong to;

6: list-of-index-and-counts -- the rest of the line contains space separated index-count pairs, where a index-count pair is in format of "index:count", E.g., "index1:count1 index2:count2" (extracted from the "text" field in the json format from Twitter)


# Dependencies:
Please install the following python libraries:

numpy version 1.13.3

theano version 1.0.4+12.g93e8180bf (dev branch for GPU compatibility)

# Reproduce the experimental results
Run script "model/Main_BU_RvNN.py" for bottom-up recursive model or "model/Main_TD_RvNN.py" for up-down recursive model.

Alternatively, you can change the "obj" parameter and "fold" parameter to set the dataset and each fold.

#If you find this code useful, please let us know and cite our paper.
