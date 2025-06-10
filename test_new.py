import numpy as np
import data_generator as dg
import find_ancestor as fa
import utils
import find_skeleton as fs
import copy

from scipy.stats import pearsonr, shapiro
import matplotlib.pyplot as plt
import draw

import causaldag as cd
import lingam

import networkx as nx

import write_excel as we

import orientation as ori

import PC_LiNGAM as PL

import time


DAG_temp = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)

# node_list_1 = [4, 5, 2]
# 
# res_1 = utils.get_subCPDAG(DAG_temp, node_list_1)

# print(res_1)

node_list_2 = [1, 2, 3, 4, 5, 6]

# res_2 = utils.get_subCPDAG(DAG_temp, node_list_2)

# print(res_2)


sub_CPDAG = utils.get_subCPDAG(DAG_temp, node_list_2)

# 获取重新编号后的componet
componets = utils.get_all_connected_componets(sub_CPDAG, node_list_2)

print(componets)