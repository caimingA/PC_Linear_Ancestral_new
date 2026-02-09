import numpy as np
import causaldag as cd
from scipy import stats
import networkx as nx
import copy
from lingam.hsic import hsic_test_gamma

from sklearn.linear_model import Lasso

import PC_LiNGAM as PL
# from pyAgrum import MixedGraph, MeekRules
# import scipy.stats as stats

from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils import Meek
from sklearn.linear_model import LinearRegression


# def matrix_to_edge(matrix):
#     node_num = len(matrix)
#     arcs = list()
#     for i in range(node_num):
#         for j in range(node_num):
#             if i == j:
#                 continue
#             else:
#                 if matrix[i][j]: #j -> i
#                     arcs.append(tuple([j, i]))
#     return arcs

def matrix_to_edge(matrix):
    node_num = len(matrix)
    arcs = [(j, i) for i in range(node_num) for j in range(node_num) if matrix[i][j]]
    return arcs

def edge_to_matrix(arcs, node_num):
    matrix = np.zeros((node_num, node_num))
    for fs in arcs:
        temp = list()
        for index in fs:
            temp.append(index) 
        j = temp[0]
        i = temp[1]
        matrix[i][j] = 1    
    return matrix

# def edge_to_matrix(arcs, node_num):
#     matrix = np.zeros((node_num, node_num), dtype=int)
#     for j, i in arcs:
#         matrix[i][j] = 1
#     return matrix


# 得到的v-structure是可识别的v-structure
def get_CPDAG(matrix):
    node_num = len(matrix)
    CPDAG = np.zeros((node_num, node_num))
    arcs = matrix_to_edge(matrix)
    g = cd.DAG(arcs=arcs)
    colliders = g.vstructures()
    cpdag = g.cpdag()
    edges = cpdag.edges
    direct_edges = cpdag.arcs
    # # print(edges)
    # # print(direct_edges)
    for fs in direct_edges:
        temp = list()
        for index in fs:
            temp.append(index) 
        j = temp[0]
        i = temp[1]
        CPDAG[i][j] = 1
        CPDAG[j][i] = -1
    for fs in edges:
        # # print(fs)
        temp = list()
        for index in fs:
            temp.append(index) 
        j = temp[0]
        i = temp[1]
        if CPDAG[i][j]:
            continue
        else:
            CPDAG[i][j] = -1
            CPDAG[j][i] = -1    
    return CPDAG, colliders


# 返回为识别出来的点和v-structure中间节点的后代节点
def get_descendants_of_all_colliders(CPDAG, collider_midnode_list):
    # 仅保留有向边，0代表无边或无向边，1代表有向边
    CPDAG_only_directed_edges = np.where(CPDAG == 1, 0, CPDAG)
    
    # 创建有向图
    G = nx.DiGraph(CPDAG_only_directed_edges)
    
    except_list = set(collider_midnode_list)
    collider_descendants_dict = {}
    
    for mid_node in collider_midnode_list:
        descendants_list = nx.descendants(G, mid_node)
        collider_descendants_dict[mid_node] = list(descendants_list)
        except_list.update(descendants_list)
    
    return list(except_list), collider_descendants_dict


def get_descendants_of_one_node(G, node):
    descendants = list(nx.descendants(G, node))
    return descendants
    

# def find_all_colliders(CPDAG):
#     node_num = len(CPDAG)
#     collider_list = list()


# def get_junction_tree(matrix):
#     G = 


def is_linear(X, Y, alpha):
    _, p = stats.pearsonr(X, Y)
    return p < alpha


def get_residual(xi, xj):
    return xi - (np.cov(xi, xj)[0, 1] / np.var(xj)) * xj


def is_independent(X, Y, alpha):
    _, p = hsic_test_gamma(X, Y, bw_method="mdbs")
    # print(p)
    return p > alpha


# 高斯变量的情况下返回True，否则返回False
def is_Gaussian(X, alpha):
    _, p = stats.shapiro(X)
    # _, p = stats.normaltest(X)
    # print(p)
    return p > alpha


# def do_Meek_rule(CPDAG):
#     node_num = len(CPDAG)
#     M = MixedGraph()
#     for i in range(node_num):
#         M.addNodeWithId(i)    
#     for i in range(node_num):
#         for j in range(i+1, node_num):
#             if CPDAG[j][i] == 1 and CPDAG[i][j] == -1:
#                 M.addArc(i, j)
#             if CPDAG[j][i] == -1 and CPDAG[i][j] == -1:
#                 M.addEdge(i, j)
    
#     Meek = MeekRules()
#     # print()
#     # CPDAG_Meek = Meek.propagateToCPDAG(M)
#     CPDAG_Meek = Meek.propagate(M)

#     return CPDAG_Meek.adjacencyMatrix
    
#     # return M.adjacencyMatrix()


def get_unfinished_edges_nodes(CPDAG):
    node_num = len(CPDAG)
    unfinished_edges = []
    finished_edges = []
    unfinished_nodes = set()
    all_nodes = set(range(node_num))
    
    for i in range(node_num):
        for j in range(i + 1, node_num):
            if CPDAG[i][j] == -1 and CPDAG[j][i] == -1:
                unfinished_edges.append([i, j])
                unfinished_nodes.update({i, j})
            else:
                finished_edges.append([i, j])
    
    finished_nodes = all_nodes - unfinished_nodes

    return np.array(sorted(list(unfinished_nodes))), unfinished_edges, np.array(sorted(list(finished_nodes))), finished_edges

def do_Meek_rule(CPDAG):
    node_num = len(CPDAG)
    cg = CausalGraph(node_num)

    # # print(cg.G.graph)
    
    for i in range(node_num):
        for j in range(node_num):
            if CPDAG[j][i] == 1 and CPDAG[i][j] == -1:
                cg.G.graph[j, i] = 1
                cg.G.graph[i, j] = -1
            if CPDAG[j][i] == 0 and CPDAG[i][j] == 0:
                cg.G.graph[j, i] = 0
                cg.G.graph[i, j] = 0
            if CPDAG[j][i] == -1 and CPDAG[i][j] == -1:
                cg.G.graph[j, i] = -1
                cg.G.graph[i, j] = -1
    # # print(cg.G.graph)
    
    cg = Meek.meek(cg) 

    CPDAG_after =  cg.G.graph
    CPDAG_after = CPDAG_after.astype(np.float64)
            
    return CPDAG_after


def get_True_CPDAG(DAG, change_list):
    node_num = len(DAG)
    CPDAG, colliders = get_CPDAG(DAG)

    # print("CPDAG: \n", CPDAG)
    unfinished_edge = list()

    for i in range(node_num):
        for j in range(i+1, node_num):
            if CPDAG[i][j] == -1 and CPDAG[j][i] == -1:
                unfinished_edge.append([i, j])

    for c in unfinished_edge:
        i = c[0]
        j = c[1]

        if i in change_list or j in change_list:
            if DAG[i][j] == 1:
                CPDAG[i][j] = 1
            if DAG[j][i] == 1:
                CPDAG[j][i] = 1
            
    CPDAG = do_Meek_rule(CPDAG)

    return CPDAG


def evaluate(CPDAG, true_CPDAG):
    # # print("inner CPDAG", CPDAG)
    node_num = len(CPDAG[0])
    
    count_none = 0
    count_undirected = 0
    count_large_to_small = 0
    count_small_to_large = 0

    res_matrix = np.zeros((4, 4))

    for i in range(node_num): # j > i
        for j in range(i+1, node_num):
            if CPDAG[i][j] == 0 and CPDAG[j][i] == 0:
                # count_none += 1
                if true_CPDAG[i][j] == 0 and true_CPDAG[j][i] == 0:
                    res_matrix[0][0] += 1
                if true_CPDAG[i][j] == -1 and true_CPDAG[j][i] == -1:
                    res_matrix[0][1] += 1
                if true_CPDAG[i][j] == 1 and true_CPDAG[j][i] == -1: # j -> i
                    res_matrix[0][2] += 1
                if true_CPDAG[i][j] == -1 and true_CPDAG[j][i] == 1:
                    res_matrix[0][3] += 1        
            if CPDAG[i][j] == -1 and CPDAG[j][i] == -1:
                if true_CPDAG[i][j] == 0 and true_CPDAG[j][i] == 0:
                    res_matrix[1][0] += 1
                if true_CPDAG[i][j] == -1 and true_CPDAG[j][i] == -1:
                    res_matrix[1][1] += 1
                if true_CPDAG[i][j] == 1 and true_CPDAG[j][i] == -1: # j -> i
                    res_matrix[1][2] += 1
                if true_CPDAG[i][j] == -1 and true_CPDAG[j][i] == 1:
                    res_matrix[1][3] += 1
            if CPDAG[i][j] == 1 and CPDAG[j][i] == -1: # j -> i
                if true_CPDAG[i][j] == 0 and true_CPDAG[j][i] == 0:
                    res_matrix[2][0] += 1
                if true_CPDAG[i][j] == -1 and true_CPDAG[j][i] == -1:
                    res_matrix[2][1] += 1
                if true_CPDAG[i][j] == 1 and true_CPDAG[j][i] == -1: # j -> i
                    res_matrix[2][2] += 1
                if true_CPDAG[i][j] == -1 and true_CPDAG[j][i] == 1:
                    res_matrix[2][3] += 1
            if CPDAG[i][j] == -1 and CPDAG[j][i] == 1:
                if true_CPDAG[i][j] == 0 and true_CPDAG[j][i] == 0:
                    res_matrix[3][0] += 1
                if true_CPDAG[i][j] == -1 and true_CPDAG[j][i] == -1:
                    res_matrix[3][1] += 1
                if true_CPDAG[i][j] == 1 and true_CPDAG[j][i] == -1: # j -> i
                    res_matrix[3][2] += 1
                if true_CPDAG[i][j] == -1 and true_CPDAG[j][i] == 1:
                    res_matrix[3][3] += 1
    
    # for i in range(node_num): # j > i
    #     for j in range(i+1, node_num):
    #         if CPDAG[i][j] == 0 and CPDAG[j][i] == 0:
    #             count_none += 1
    #         if CPDAG[i][j] == -1 and CPDAG[j][i] == -1:
    #             count_undirected += 1
    #         if CPDAG[i][j] == 1 and CPDAG[j][i] == -1: # j -> i
    #             count_large_to_small += 1
    #         if CPDAG[i][j] == -1 and CPDAG[j][i] == 1:
    #             count_small_to_large += 1
    
    # return count_none, count_undirected, count_large_to_small, count_small_to_large
    return res_matrix


def get_all_connected_componets(CPDAG, true_node_list):
    # 创建无向图，只保留有边的部分，忽略方向性
    G = nx.Graph()
    G.add_nodes_from(range(len(CPDAG)))
    non_zero_indices = np.where(CPDAG != 0)
    edges = zip(non_zero_indices[0], non_zero_indices[1])
    G.add_edges_from(edges)
    
    # 获取所有的连通分量
    connected_components = list(nx.connected_components(G))

    # 构建连通分量的真节点列表
    components_list = [
        [true_node_list[i] for i in sorted(comp)]
        for comp in connected_components
    ]

    return components_list


def get_all_roots(DAG, target):
    # 创建无向图，只保留有边的部分，忽略方向性
    # G = nx.DiGraph()
    # G.add_nodes_from(range(len(CPDAG)))
    # non_zero_indices = np.where(CPDAG != 0)
    # edges = zip(non_zero_indices[0], non_zero_indices[1])
    # print(edges)
    # G.add_edges_from(edges)

    G = nx.from_numpy_array(DAG.T, create_using=nx.DiGraph())
    # print(nx.adjacency_matrix(G))
    roots = list()
    for node in range(len(DAG)):
        if G.in_degree(node) == 0:
            roots.append(node)
    # roots = [node for node, in_degree in G.in_degree() if in_degree == 0]
    # print(roots)
    reachable_roots = [root for root in roots if nx.has_path(G, root, target)]

    return reachable_roots



def extract_data_unfinished(data, CPDAG, unfinished_nodes, finished_nodes):
    reg = LinearRegression(fit_intercept=False)
    data_unfinished = []

    # 将 CPDAG 转换为布尔矩阵，便于快速检查父节点关系
    parent_relation = (CPDAG == 1) & (CPDAG.T == -1)

    for i in unfinished_nodes:
        # 找到所有与节点 i 相关的父节点
        parents_indices = np.where(parent_relation[i])[0]
        parents_indices = [j for j in parents_indices if j in finished_nodes]

        if len(parents_indices) != 0:
            parents_data = data[:, parents_indices]
            reg.fit(parents_data, data[:, i])
            residuals = data[:, i] - reg.predict(parents_data)
            data_unfinished.append(residuals)
        else:
            data_unfinished.append(data[:, i])

    return np.vstack(data_unfinished).T


def extract_data_unfinished_2(data, CPDAG, unfinished_nodes, finished_nodes):
    # reg = LinearRegression(fit_intercept=False)
    data_unfinished_residual = list()
    # parent_list = list()

    parent_relation = (CPDAG == 1) & (CPDAG.T == -1)
    for i in unfinished_nodes:
        parents_indices = np.where(parent_relation[i])[0]
        parents_indices += [j for j in parents_indices if j in finished_nodes]
    lasso = Lasso(0.1)
    parents_data = data[:, parents_indices]
    unfinished_data = data[:, unfinished_nodes]
    
    lasso.fit(parents_data,  unfinished_data)
    data_unfinished_residual = unfinished_data - lasso.predict(parents_data)

    return data_unfinished_residual
    # 将 CPDAG 转换为布尔矩阵，便于快速检查父节点关系
    # parent_relation = (CPDAG == 1) & (CPDAG.T == -1)

    # for i in unfinished_nodes:
    #     # 找到所有与节点 i 相关的父节点
    #     parents_indices = np.where(parent_relation[i])[0]
    #     parents_indices = [j for j in parents_indices if j in finished_nodes]

    #     if len(parents_indices) != 0:
    #         parents_data = data[:, parents_indices]
    #         reg.fit(parents_data, data[:, i])
    #         residuals = data[:, i] - reg.predict(parents_data)
    #         data_unfinished.append(residuals)
    #     else:
    #         data_unfinished.append(data[:, i])

    # return np.vstack(data_unfinished).T

# 真实编号
def get_subCPDAG(CPDAG, node_list):
    return CPDAG[np.ix_(node_list, node_list)]


# def map_CPDAG_to_subCPDAG(X_data, CPDAG, componets, true_nodes):
#     mapping_dict = {node: idx for idx, node in enumerate(true_nodes)}
#     mapping_list = []

#     for comp in componets:
#         # x_data = X_data[count]
#         node_num = len(comp)
#         node_list = list()
#         sub_CPDAG = np.zeros((node_num, node_num))
#         # sub_CPDAG = CPDAG[np.ix_(comp, comp)]
#         data_unfinished = list()
#         for i in range(node_num):
#             node_list.append(comp[i])
#             # node_list.append(true_nodes[comp[i]])
#             data_unfinished.append(X_data[:, mapping_dict[comp[i]]])
#             for j in range(i+1, node_num):
#                 sub_CPDAG[i][j] = CPDAG[mapping_dict[comp[i]]][mapping_dict[comp[j]]]
#                 sub_CPDAG[j][i] = CPDAG[mapping_dict[comp[j]]][mapping_dict[comp[i]]]            
#         mapping_list.append([node_list, sub_CPDAG, np.array(data_unfinished).T])
#     return mapping_list


def map_CPDAG_to_subCPDAG(X_data, CPDAG, components, true_nodes):
    mapping_dict = {node: idx for idx, node in enumerate(true_nodes)}
    mapping_list = []

    for comp in components:
        node_indices = [mapping_dict[node] for node in comp]
        sub_CPDAG = CPDAG[np.ix_(node_indices, node_indices)]
        sub_data = X_data[:, node_indices]
        mapping_list.append([comp, sub_CPDAG, sub_data])

    return mapping_list


# input is MAG
def get_new_vstructure(matrix, colliders):
    node_num = len(matrix[0])
    new_vstructures_list = list()
    new_vstrcutures_edge_list = list()
    for i in range(node_num):
        if i in colliders:
            continue
        else:
            direct_edge_list = list()
            for j in range(node_num):
                if matrix[i][j] == 1 and matrix[j][i] == -1:
                    for k in direct_edge_list:
                        if matrix[k][j] == 0:
                            if i not in new_vstructures_list:
                                new_vstructures_list.append(i)
                            if [k, i] not in new_vstrcutures_edge_list:
                                new_vstrcutures_edge_list.append([k, i])
                            if [j, i] not in new_vstrcutures_edge_list:
                                new_vstrcutures_edge_list.append([j, i])
                    direct_edge_list.append(i)
    return new_vstructures_list, new_vstrcutures_edge_list


def detect_expections(CPDAG, collider_old):
    node_num = len(CPDAG)
    DAG_temp = np.where(CPDAG == 1.0, 1.0, 0)
    loop_flag = PL.topological_sort(DAG_temp)
    if loop_flag == 0:
        # print("loop")
        # input()
        return 1
    arcs = matrix_to_edge(DAG_temp)
    g = cd.DAG(arcs=arcs)
    collider_new = g.vstructures()
    
    
    
    for vs in collider_new:
        vs_inverse = (vs[2], vs[1], vs[0])
        if vs in collider_old or vs_inverse in collider_old:
            pass
        else:
            # print("VS")
            # print("old: ", collider_old)
            # print("new: ", collider_new)
            # input()
            return 2    
    return 0

    # new_vstructures_list = list()
    # new_vstrcutures_edge_list = list()

    # for i in range(node_num):


def detect_exceptions_2(CPDAG, collider_old):
    node_num = len(CPDAG)
    DAG_temp = np.where(CPDAG == 1.0, 1.0, 0.0)
    topo_order = PL.topological_sort_2(DAG_temp)
    # while topo_order != node_num: 
    if len(topo_order) != node_num:
        # print("loop")
        unfinished_node = sorted(list(set([i for i in range(node_num)]) - set(topo_order)))
        unfinished_num = len(unfinished_node)
        # print(unfinished_node)
        for i in range(unfinished_num):
            for j in range(i+1, unfinished_num):
                if CPDAG[unfinished_node[j]][unfinished_node[i]] != 0.0:
                    CPDAG[unfinished_node[j]][unfinished_node[i]] = 1.0
                    CPDAG[unfinished_node[i]][unfinished_node[j]] = -1.0

        # input()
    #     return 1
    DAG_temp = np.where(CPDAG == 1.0, 1.0, 0.0)
    arcs = matrix_to_edge(DAG_temp)
    g = cd.DAG(arcs=arcs)
    collider_new = g.vstructures()
    for vs in collider_new:
        vs_inverse = (vs[2], vs[1], vs[0])
        if vs in collider_old or vs_inverse in collider_old:
            pass
        else:
            # print("VS")
            visit_list = list()
            root_nodes_list = get_all_roots(DAG_temp, vs[1])
            # print("roots: ", root_nodes_list)
            root_node = np.min(root_nodes_list)
            i_pos = root_node
            
            while i_pos != None:
                next_list = list()
                # print(i_pos)
                # print(next_list)
                visit_list.append(i_pos)
                for j in range(node_num):
                    # print("adj: ", j, "and ", CPDAG[j][i_pos])
                    if CPDAG[j][i_pos] != 0 and j not in visit_list:
                        flag_temp = True
                        for vs_old in collider_old:
                            if vs_old[0] == j and vs_old[1] == i_pos or vs_old[2] == j and vs_old[1] == i_pos:
                                # print(vs_old)
                                # print(j, "and ", i_pos)
                                flag_temp = False
                                # break
                        if flag_temp:
                            # print("change: ", j)
                            CPDAG[j][i_pos] = 1.0
                            CPDAG[i_pos][j] = -1.0
                            next_list.append(j)
                if len(next_list) != 0:
                    i_pos = np.min(next_list)
                else:
                    i_pos = None
            # if vs[0] < vs[2]:
            #     CPDAG[vs[2]][vs[1]] = 1
            #     CPDAG[vs[1]][vs[2]] = -1
            # print("old: ", collider_old)
            # print("new: ", collider_new)
            # input()
            # return 2    
    return CPDAG


# def handle_expections(CPDAG, )


def match_to_MEC(CPDAG, MEC, colliders):
    new_vstructures_list, new_vstrcutures_edge_list = get_new_vstructure(CPDAG, colliders)
    if len(new_vstructures_list) == 0:
        return CPDAG
    else:
        for c in new_vstrcutures_edge_list:
            i = c[1]
            j = c[0] # j -> i            
            CPDAG[j][i] = 1
            CPDAG[i][j] = -1
            nvl, nvel = get_new_vstructure(CPDAG, colliders)
            if len(nvel) < len(new_vstrcutures_edge_list):
                pass
            else:
                CPDAG[j][i] = -1
                CPDAG[i][j] = 1    
    return CPDAG

# def v_structure_find(CPDAG):