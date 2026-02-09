import numpy as np
import find_ancestor as fa
import utils
import find_skeleton as fs
import copy
import itertools
import causaldag as cd
from sklearn.linear_model import LinearRegression
import draw
import time


def topological_sort(adj_matrix):
    n = adj_matrix.shape[0]
    in_degree = np.sum(adj_matrix, axis=0)
    queue = [i for i in range(n) if in_degree[i] == 0]
    top_order = list()

    while queue:
        node = queue.pop(0)
        top_order.append(node)
        for i in range(n):
            if adj_matrix[node][i] == 1:
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    queue.append(i)

    if len(top_order) != n:
        return 0
    return top_order


# def generate_potential_DAGs_according_to_CPDAG(CPDAG, colliders):
#     node_num = len(CPDAG)
#     unfinished_edge = list()
#     potential_DAGs_list = list()
#     all_DAGs_list = list()
    
#     for i in range(node_num):
#         for j in range(i+1, node_num):
#             if CPDAG[i][j] == -1.0 and CPDAG[j][i] == -1.0:
#                 unfinished_edge.append([i, j])

#     for num in range(len(unfinished_edge)+1):
#         for item in itertools.combinations(unfinished_edge, num):
#             DAG_temp = np.where(CPDAG == 1.0, 1.0, 0.0)
#             for edge in unfinished_edge:
#                 i = edge[0]
#                 j = edge[1]
#                 if edge in item:
#                     DAG_temp[j][i] = 1
#                 else:
#                     DAG_temp[i][j] = 1
#             if topological_sort(DAG_temp) != 0:
#                 all_DAGs_list.append(DAG_temp)
#                 arcs = utils.matrix_to_edge(DAG_temp)
#                 g = cd.DAG(arcs=arcs)
#                 colliders_temp = g.vstructures()
#                 colliders_temp_set = set(colliders_temp)
#                 colliders_old_set = set(colliders)
                
#                 if colliders_temp_set == colliders_old_set:
#                     potential_DAGs_list.append(DAG_temp)

#     if len(potential_DAGs_list) == 0:
#         potential_DAGs_list = all_DAGs_list
#     return potential_DAGs_list

def generate_potential_DAGs_according_to_CPDAG(CPDAG, colliders):
    node_num = len(CPDAG)
    unfinished_edge = list()
    potential_DAGs_list = list()
    all_DAGs_list = list()
    # potential_causalDAGs_list = list()
    # map_dict = dict()
    
    for i in range(node_num):
        # map_dict[X_ng[i]] = i
        # start = X_ng[i]
        for j in range(i+1, node_num):
            # end = X_ng[j]
            if CPDAG[i][j] == -1.0 and CPDAG[j][i] == -1.0:
                unfinished_edge.append([i, j])
                # DAG_temp = 

    for num in range(len(unfinished_edge)+1):
        for item in itertools.combinations(unfinished_edge, num):
            # print(item)
            DAG_temp = np.where(CPDAG == 1.0, 1.0, 0.0)
            for edge in unfinished_edge:
                i = edge[0]
                j = edge[1]
                if edge in item:
                    DAG_temp[j][i] = 1
                else:
                    DAG_temp[i][j] = 1
            if topological_sort(DAG_temp) != 0:
                all_DAGs_list.append(DAG_temp)
                arcs = utils.matrix_to_edge(DAG_temp)
                g = cd.DAG(arcs=arcs)
                colliders_temp = g.vstructures()
                colliders_temp_copy = list()
                colliders_old_copy = list()
                for vs in colliders:
                    colliders_old_copy.append(vs)
                    colliders_old_copy.append((vs[2], vs[1], vs[0]))
                for vs in colliders_temp:
                    colliders_temp_copy.append(vs)
                    colliders_temp_copy.append((vs[2], vs[1], vs[0]))
                
                if set(colliders_temp_copy) == set(colliders_old_copy):
                    potential_DAGs_list.append(DAG_temp)
                    # potential_causalDAGs_list.append(g)
    # for dag in potential_DAGs_list:
    #     print(dag)
    if len(potential_DAGs_list) == 0:
        potential_DAGs_list = all_DAGs_list
    return potential_DAGs_list


def get_residual_for_one_DAG(data, DAG):
    node_num = len(DAG)
    arcs = utils.matrix_to_edge(DAG)
    g = cd.DAG(arcs=arcs)
    residuals_list = list()
    
    reg = LinearRegression(fit_intercept=False)
    for i in range(node_num):
        ancestors_list = g.ancestors_of(i)
        if len(ancestors_list) == 0:
            residuals_list.append(data[:, i])
        else:
            ancestors_data_list = list()
            for anc in ancestors_list:
                ancestors_data_list.append(data[:, anc])
            ancestors_data_list = np.array(ancestors_data_list).T

            reg.fit(ancestors_data_list, data[:, i])
            predicted = reg.predict(ancestors_data_list)
            residual = data[:, i] - predicted

            residuals_list.append(residual)
    
    return residuals_list


def sorce_function(residuals_list):
    node_num = len(residuals_list)
    residual_fabs = list()
    for i in range(node_num):
        residual_fabs.append(np.fabs(residuals_list[i]))
    residuals_means = list()
    for i in range(node_num):
        residuals_means.append(np.mean(residual_fabs[i]))
    
    score = 0
    for i in range(node_num):
        score += (residuals_means[i] - np.sqrt(2/np.pi))**2
    
    return score 


def select_best_DAG(data, DAG_list):
    DAG_num = len(DAG_list)
    print("The number of DAGs: ", DAG_num)
    score = -1
    best_index = -1
    best_residuals_list = list()
    
    for i in range(DAG_num):
        residuals_list = get_residual_for_one_DAG(data, DAG_list[i])
        score_now = sorce_function(residuals_list)
        
        if score_now > score:
            score = score_now
            best_index = i
            best_residuals_list = residuals_list
    return DAG_list[best_index], best_residuals_list


def delete_direction(CPDAG, DAG, residual_list, s_alpha):
    node_num = len(CPDAG)
    DAG_result = np.copy(DAG)
    unfinished_edge = list()
    
    for i in range(node_num):
        for j in range(i+1, node_num):
            if CPDAG[i][j] == -1 and CPDAG[j][i] == -1:
                unfinished_edge.append([i, j])

    print(f"unfinished_edge: {unfinished_edge}")
    for edge in unfinished_edge:
        i = edge[0]
        j = edge[1]

        if utils.is_Gaussian(residual_list[i], s_alpha) and utils.is_Gaussian(residual_list[j], s_alpha):
            DAG_result[i][j] = -1
            DAG_result[j][i] = -1
    
    for i in range(node_num):
        for j in range(node_num):
            if DAG_result[i][j] == 1:
                DAG_result[j][i] = -1
    
    return DAG_result


def PC_LiNGAM(data, DAG, s_alpha):
    node_num = len(data[0])
    DAG_temp = np.where(DAG == 1.0, 1.0, 0.0)
    CPDAG_temp, colliders = utils.get_CPDAG(DAG_temp)
    DAG_list = generate_potential_DAGs_according_to_CPDAG(CPDAG_temp, colliders)
    best_DAG, best_residuals = select_best_DAG(data, DAG_list)
    CPDAG_res = delete_direction(CPDAG_temp, best_DAG, best_residuals, s_alpha)
    
    return CPDAG_res

def PC_LiNGAM_CPDAG(data, DAG, s_alpha):
    node_num = len(data[0])
    DAG_temp = np.where(DAG == 1.0, 1.0, 0.0)
    CPDAG_temp, colliders = utils.get_CPDAG(DAG_temp)
    DAG_list = generate_potential_DAGs_according_to_CPDAG(DAG, colliders)
    best_DAG, best_residuals = select_best_DAG(data, DAG_list)
    # CPDAG_res = delete_direction(CPDAG_temp, best_DAG, best_residuals, s_alpha)
    CPDAG_res = delete_direction(DAG, best_DAG, best_residuals, s_alpha)
    
    return CPDAG_res


def PC_LiNGAM_multithread(experiment_set):
    data = experiment_set[0]
    DAG = experiment_set[1]
    s_alpha = experiment_set[2]

    node_num = len(data[0])
    start_time = time.process_time()
    CPDAG, colliders = utils.get_CPDAG(DAG)
    DAG_list = generate_potential_DAGs_according_to_CPDAG(CPDAG, colliders)
    best_DAG, best_residuals = select_best_DAG(data, DAG_list)
    CPDAG_res = delete_direction(CPDAG, best_DAG, best_residuals, s_alpha)
    CPDAG_res = utils.do_Meek_rule(np.array(CPDAG_res.astype(np.float64)))

    end_time = time.process_time()
    
    return CPDAG_res, end_time - start_time
