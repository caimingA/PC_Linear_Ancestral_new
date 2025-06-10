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


# print(type({(2, 1), (2, 4), (2, 3), (1, 3), (3, 4)}))
# print(type((2, 1)))
# # g = cd.DAG(arcs={(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)})
# g = cd.DAG(arcs={(2, 1), (2, 4), (2, 3), (1, 3), (3, 4)})
# cpdag = g.cpdag()
# print(cpdag.edges)
# print(cpdag.arcs)


# DAG_temp = np.array([
#         [0.0, 0.0],
#         [1.0, 0.0]
#     ]
# )


# DAG_temp = np.array([
#         [0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0],
#         [1.0, 1.0, 0.0],
#     ]
# )

# DAG_temp = np.array([
#         [0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0],
#     ]
# )

# DAG_temp = np.array([
#         [0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0],
#         [1.0, 1.0, 0.0, 0.0],
#         [1.0, 1.0, 1.0, 0.0],
#     ]
# )

# 5 complete
# DAG_temp = np.array([
#         [0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 1.0, 0.0, 0.0, 0.0],
#         [1.0, 1.0, 1.0, 0.0, 0.0],
#         [1.0, 1.0, 1.0, 1.0, 0.0]
#     ]
# )

# 5 tree
# DAG_temp = np.array([
#         [0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 1.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 1.0, 0.0]
#     ]
# )



# 6
# DAG_temp = np.array([
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
#         [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
#         [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
#     ]
# )

DAG_temp = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    ]
)

# DAG_temp = np.array([
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     ]
# )

# DAG_temp = np.array([
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
#     ]
# )

# 10
# DAG_temp = np.array([
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
#         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
#         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
#     ]
# )
# temp = nx.DiGraph()
# G = nx.from_numpy_array(DAG_temp)
# nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
# plt.show()
# DAG_temp = np.array([
#         [0, 0, 0, 0, 0, 0],
#         [1, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0],
#         [0, 1, 0, 0, 1, 0],
#         [0, 1, 1, 0, 0, 0],
#         [0, 0, 1, 0, 1, 0]
#     ]
# )

# edges = utils.matrix_to_edge(DAG_temp)

# node_num = 5
# print(np.random.get_state()[1][0])
np.random.seed(2147483648)
print(DAG_temp)

node_num = len(DAG_temp[0])
sample_size = 1000

# DAG_temp = dg.lower_triangle_graph(node_num, node_num - 1, 1)

B_temp = dg.get_B_0(DAG_temp)
print(B_temp)
# # draw.draw_undirected_graph(B_temp)
# change_index = [1, 3]
change_index = [0, 1, 4]
# change_index = [2]

# draw.draw_directed_graph(DAG_temp)
draw.draw_directed_graph(B_temp)

# change_index = np.random.randint(0, node_num)
# change_index = [1, 4]
# change_index = [0]
print("change node: ", change_index)
noisy_temp = dg.get_noisy_Mix(sample_size, DAG_temp, [change_index])
# noisy_temp = dg.get_noisy_Gaussian(sample_size, DAG_temp)

# B_temp = np.array([
#         [0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0],
#         [1.0, 1.0, 0.0],
#     ]
# )


x_list_temp = dg.get_x(DAG_temp, noisy_temp, B_temp).T

print(x_list_temp.shape)

# we.write_x_excel(x_list_temp, "test_5")

# y_list_temp = copy.deepcopy(x_list_temp)
# noisy_0 = np.random.random(size = (1000))*2
# noisy_0 = np.where(noisy_0 > 1.0, noisy_0+1.0, noisy_0-3.0)

# noisy_1 = np.random.random(size = (1000))*2
# noisy_1 = np.where(noisy_1 > 1.0, noisy_1+1.0, noisy_1-3.0)

# print(noisy_0.shape)
# y_list_temp[:, 0] += noisy_0
# y_list_temp[:, 1] += noisy_1


# for i in range(node_num):
#     for j in range(i+1, node_num):
#         print("=================")
#         print(i, "and ", j)
#         res_flag = fa.get_ancestor_pairwise_HSIC(x_list_temp[:,i], x_list_temp[:,j], 0.01, 0.01)
#         if res_flag == 0:
#             print("no relationship")
#         if res_flag == 1:
#             print(j, "--> ", i)
#         if res_flag == 2:
#             print(i, "--> ", j)
#         if res_flag == 3:
#             print(i, "and ", j, "are Gaussian")
#         if res_flag == 4:
#             print("have common ancestor")


# print(x_list_temp.shape)
# print(B_temp)


# for i in range(node_num):
#     plt.hist(noisy_temp[: , i], bins=50)
#     plt.title(str(i) + " distribution")
#     plt.show()

# for i in range(node_num):
#     plt.hist(x_list_temp[: , i], bins=50)
#     plt.title(str(i) + " distribution")
#     plt.show()


# gaussian_list, nonGaussian_list = fs.extract_Gaussian_and_nonGaussian(x_list_temp, 0.05)
# print(gaussian_list)
# print(nonGaussian_list)

# residual = fs.regress_nonGaussian_on_Gaussian(x_list_temp, gaussian_list, nonGaussian_list)
# print(residual.shape)

# model_L = lingam.DirectLiNGAM()

# model_L.fit(x_list_temp)
# m_L = model_L.adjacency_matrix_

# print("0 and 1: ", fa.get_ancestor_pairwise_HSIC(x_list_temp[:, 0], x_list_temp[:, 1], 0.01, 0.05))
# print("0 and 2: ", fa.get_ancestor_pairwise_HSIC(x_list_temp[:, 0], x_list_temp[:, 2], 0.01, 0.05))
# print("1 and 2: ", fa.get_ancestor_pairwise_HSIC(x_list_temp[:, 1], x_list_temp[:, 2], 0.01, 0.05))
# print("1 and 2: ", fa.get_ancestor_pairwise_HSIC(x_list_temp[:, 2], x_list_temp[:, 3], 0.01, 0.05))

# print("y: ", fa.get_ancestor_pairwise_HSIC(y_list_temp[:, 0], y_list_temp[:, 1], 0.01, 0.01))


# print(utils.get_CPDAG(DAG_temp))
# model_RCD = lingam.RCD()
# model_RCD.fit(x_list_temp)
# m_RCD = model_RCD.adjacency_matrix_

# print(m_RCD)
# print(model_RCD.ancestors_list_)

# print(B_temp)
# print(model_L.causal_order_)
# print(m_L)

# draw.draw_directed_graph(m_L)
# anc_dict = fa.get_ancestor_tripe_HSIC(x_list_temp, 0.01)
# print(anc_dict)

# CPDAG = dM.find_MEC(x_list_temp, 0.05, "kci")

# print(CPDAG)

# # draw.draw_directed_graph(CPDAG)
# # CPDAG = np.array([
# #         [0.0, -1.0, 0.0],
# #         [-1.0, 0.0, -1.0],
# #         [0.0, -1.0, 0.0],
# #     ]
# # )

####################
# CPDAG, colliders = utils.get_CPDAG(DAG_temp)

# # # # print(DAG_temp)
# # # # print(CPDAG)
# draw.draw_CPDAG(CPDAG)
# print("colliders: ", colliders)
# collider_dict, collider_midnode_list = fs.extract_colliders(colliders)
# print(collider_dict)
# print(collider_midnode_list)

# exc_res, des_res = utils.get_descendants_of_all_colliders(CPDAG, collider_midnode_list)
# print(exc_res)
# print(des_res)

# des = utils.get_descendants_of_one_node(CPDAG, collider_midnode_list[0])
# print(des)
####################

# fs.extract_skeleton_DFS(CPDAG, collider_dict, collider_midnode_list)


# print(DAG_temp)

# s_alpha = 0.01

######################################################
# time_1 = time.time()
CPDAG, colliders = utils.get_CPDAG(DAG_temp)
print(CPDAG)

draw.draw_CPDAG(CPDAG)

print("=======")
# CPDAG = ori.identify_direction_begin_with_CPDAG(x_list_temp, DAG_temp, 0.01, 0.01, 0.1)
CPDAG = ori.identify_direction_begin_with_CPDAG_new(x_list_temp, DAG_temp, 0.01, 0.01, 0.001)
print(CPDAG)
# # draw.draw_CPDAG(CPDAG)
CPDAG = utils.do_Meek_rule(CPDAG)
print(CPDAG)
draw.draw_CPDAG(CPDAG)
# time_2 = time.time()

print("=======")
# # print(len(PL.generate_potential_DAGs_according_to_CPDAG(CPDAG, colliders)))

CPDAG = PL.PC_LiNGAM(x_list_temp, DAG_temp, 0.01)
print(CPDAG)
# # draw.draw_CPDAG(CPDAG)

CPDAG = utils.do_Meek_rule(CPDAG)
print(CPDAG)
draw.draw_CPDAG(CPDAG)
# time_3 = time.time()

# print("time proposed: ", time_2 - time_1)
# print("time PC_L: ", time_3 - time_2)
######################################################

# print(type(utils.get_all_connected_componets(DAG_temp)[0]))

# utils.get_all_connected_componets(CPDAG)