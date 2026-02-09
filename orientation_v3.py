import numpy as np
import find_ancestor_v2 as fa
import utils_v2 as utils
import copy
from multiprocessing import Pool
import time

import PC_LiNGAM_optimized as PL


def identify_direction_begin_with_CPDAG(X, DAG, s_alpha, l_alpha, i_alpha):
    X_data = X.copy()
    node_num = len(DAG)
    node_list = [i for i in range(node_num)]
    
    CPDAG, colliders = utils.get_CPDAG(DAG)
    collider_dict, collider_midnode_list = fa.extract_colliders(colliders)
    direct_nodes, des_res = utils.get_descendants_of_all_colliders(CPDAG, collider_midnode_list)
    
    E_di = direct_nodes
    E_ud = list(set(node_list) - set(direct_nodes))

    X_ga, X_ng = fa.extract_Gaussian_and_nonGaussian(X_data, s_alpha)

    if len(X_ng) == 0:
        return CPDAG
    
    while len(X_ng) != 0:
        for i in X_ga:
            for j in X_ng:
                if CPDAG[i][j] == -1 and CPDAG[j][i] == -1:
                    CPDAG[j][i] = 1
        
        residuals_on_Gaussian_list = fa.regress_nonGaussian_on_Gaussian(X_data, X_ga, X_ng)
        CPDAG, x_0 = fa.get_ancestor_follow_skeleton(residuals_on_Gaussian_list, CPDAG, X_ng, l_alpha, i_alpha)
        
        if len(X_ng) == 1 and x_0 == X_ng[0]:
            break
        else:
            residuals_on_x0_list = fa.regress_nonGaussian_on_x0(residuals_on_Gaussian_list, x_0, X_ng)
            X_ng = X_ng[X_ng != x_0]
            
            X_ga_temp, X_ng_temp = fa.extract_Gaussian_and_nonGaussian(residuals_on_x0_list, s_alpha)
            X_data = residuals_on_x0_list
            X_ga_next = [X_ng[i] for i in X_ga_temp]
            X_ng_next = [X_ng[i] for i in X_ng_temp]
            X_ng = X_ng_next
            X_ga = X_ga_next
    return CPDAG


def identify_direction_begin_with_CPDAG_revised(X, DAG, s_alpha, l_alpha, i_alpha):
    X_data = X.copy()
    node_num = len(DAG)
    
    CPDAG, colliders = utils.get_CPDAG(DAG)
    
    unfinished_nodes, unfinished_edges, finished_nodes, finished_edges = utils.get_unfinished_edges_nodes(CPDAG)
    
    sub_CPDAG = utils.get_subCPDAG(CPDAG, unfinished_nodes) 
    X_data = utils.extract_data_unfinished(X_data, CPDAG, unfinished_nodes, finished_nodes)
    componets = utils.get_all_connected_componets(sub_CPDAG)
    mapping_res = utils.map_CPDAG_to_subCPDAG(X_data, sub_CPDAG, componets, unfinished_nodes)
    
    while len(mapping_res):
        componet_now = mapping_res.pop()
        true_nodes_now = componet_now[0]
        sub_CPDAG_now = componet_now[1]
        X_data_now = componet_now[2] 
        
        X_ga, X_ng = fa.extract_Gaussian_and_nonGaussian(X_data_now, s_alpha)

        if len(X_ng) == 0:
            continue
        
        while len(X_ng) != 0:
            for i in X_ga:
                for j in X_ng:
                    if sub_CPDAG_now[i][j] == -1 and sub_CPDAG_now[j][i] == -1:
                        sub_CPDAG_now[j][i] = 1
            
            residuals_on_Gaussian_list = fa.regress_nonGaussian_on_Gaussian(X_data_now, X_ga, X_ng)
            
            sub_CPDAG_now_now = utils.get_subCPDAG(sub_CPDAG_now, X_ng)
            componets_now_now = utils.get_all_connected_componets(sub_CPDAG_now_now)
            mapping_res_now_now = utils.map_CPDAG_to_subCPDAG(residuals_on_Gaussian_list, sub_CPDAG_now_now, componets_now_now, X_ng)
            if len(mapping_res_now_now)!=1:
                mapping_res += mapping_res_now_now
                break
            
            sub_CPDAG_now, x_0 = fa.get_ancestor_follow_skeleton(residuals_on_Gaussian_list, sub_CPDAG_now, X_ng, l_alpha, i_alpha)
            
            if len(X_ng) == 1 and x_0 == X_ng[0]:
                break
            else:
                residuals_on_x0_list = fa.regress_nonGaussian_on_x0(residuals_on_Gaussian_list, x_0, X_ng)
                X_ng = X_ng[X_ng != x_0]
                sub_CPDAG_now_now = utils.get_subCPDAG(sub_CPDAG_now, X_ng)
                componets_now_now = utils.get_all_connected_componets(sub_CPDAG_now_now)
                mapping_res_now_now = utils.map_CPDAG_to_subCPDAG(residuals_on_x0_list, sub_CPDAG_now_now, componets_now_now, X_ng)
                if len(mapping_res_now_now)!=1:
                    mapping_res += mapping_res_now_now
                    break
                X_ga_temp, X_ng_temp = fa.extract_Gaussian_and_nonGaussian(residuals_on_x0_list, s_alpha)
                X_data_now = residuals_on_x0_list
                X_ga_next = [X_ng[i] for i in X_ga_temp]
                X_ng_next = [X_ng[i] for i in X_ng_temp]
                X_ng = X_ng_next
                X_ga = X_ga_next
                
        for i in range(len(true_nodes_now)):
            for j in range(len(true_nodes_now)):
                if sub_CPDAG_now[i][j] == 1 and sub_CPDAG_now[j][i] == -1:
                    CPDAG[true_nodes_now[i]][true_nodes_now[j]] = 1
                    CPDAG[true_nodes_now[j]][true_nodes_now[i]] = -1
    CPDAG = utils.do_Meek_rule(CPDAG)
    return CPDAG


def identify_direction_begin_with_CPDAG_new(X, DAG, s_alpha, l_alpha, i_alpha):
    X_data = X.copy()
    std = np.std(X_data, axis=0)
    std[std == 0] = 1.0
    mean = np.mean(X_data, axis=0)
    X_data = (X_data - mean) / std
    
    node_num = len(DAG)
    CPDAG = np.copy(DAG)
    
    unfinished_nodes, unfinished_edges, finished_nodes, finished_edges = utils.get_unfinished_edges_nodes(CPDAG)
    
    if len(unfinished_nodes) == 0:
        return CPDAG
    
    sub_CPDAG = utils.get_subCPDAG(CPDAG, unfinished_nodes)
    X_data = utils.extract_data_unfinished(X_data, CPDAG, unfinished_nodes, finished_nodes)
    componets = utils.get_all_connected_componets(sub_CPDAG, unfinished_nodes)
    mapping_res = utils.map_CPDAG_to_subCPDAG(X_data, sub_CPDAG, componets, unfinished_nodes)

    while len(mapping_res):
        componet_now = mapping_res.pop()
        true_nodes_now = componet_now[0]
        X_data_now = componet_now[2] 
        
        if len(true_nodes_now) == 1:
            continue
        
        X_ga, X_ng = fa.extract_Gaussian_and_nonGaussian(X_data_now, true_nodes_now, s_alpha)
    
        if len(X_ng) == 0:
            continue
        
        while len(X_ng) != 0:
            for i in X_ga:
                for j in X_ng:
                    if CPDAG[i][j] == -1 and CPDAG[j][i] == -1:
                        CPDAG[j][i] = 1
            
            if len(X_ng) == 1:
                break

            residuals_on_Gaussian_list = fa.regress_nonGaussian_on_Gaussian(X_data_now, X_ga, X_ng)

            sub_CPDAG_now_now = utils.get_subCPDAG(CPDAG, X_ng)
            componets_now_now = utils.get_all_connected_componets(sub_CPDAG_now_now, X_ng)

            mapping_res_now_now = utils.map_CPDAG_to_subCPDAG(residuals_on_Gaussian_list, CPDAG, componets_now_now, X_ng)
            if len(mapping_res_now_now)!=1:
                mapping_res += mapping_res_now_now
                break

            CPDAG, x_0 = fa.get_ancestor_follow_skeleton(residuals_on_Gaussian_list, CPDAG, X_ng, l_alpha, i_alpha)
            
            if len(X_ng) == 1:
                break
            else:
                residuals_on_x0_list = fa.regress_nonGaussian_on_x0(residuals_on_Gaussian_list, x_0, X_ng)
                X_ng = X_ng[X_ng != x_0]
                sub_CPDAG_now_now = utils.get_subCPDAG(CPDAG, X_ng)
                componets_now_now = utils.get_all_connected_componets(sub_CPDAG_now_now, X_ng)

                mapping_res_now_now = utils.map_CPDAG_to_subCPDAG(residuals_on_x0_list, CPDAG, componets_now_now, X_ng)
                if len(mapping_res_now_now)!=1:
                    mapping_res += mapping_res_now_now
                    break

                X_ga, X_ng = fa.extract_Gaussian_and_nonGaussian(residuals_on_x0_list, X_ng, s_alpha)
                X_data_now = residuals_on_x0_list

    return CPDAG


def identify_direction_begin_with_CPDAG_new_2(X, DAG, s_alpha, l_alpha, i_alpha):
    X_data = X.copy()
    std = np.std(X_data, axis=0)
    std[std == 0] = 1.0
    mean = np.mean(X_data, axis=0)
    X_data = (X_data - mean) / std
    
    node_num = len(DAG)
    CPDAG = np.copy(DAG)
    
    unfinished_nodes, unfinished_edges, finished_nodes, finished_edges = utils.get_unfinished_edges_nodes(CPDAG)
    
    if len(unfinished_nodes) == 0:
        return CPDAG
    
    sub_CPDAG = utils.get_subCPDAG(CPDAG, unfinished_nodes)
    X_data = utils.extract_data_unfinished(X_data, CPDAG, unfinished_nodes, finished_nodes)
    componets = utils.get_all_connected_componets(sub_CPDAG, unfinished_nodes)
    mapping_res = utils.map_CPDAG_to_subCPDAG(X_data, sub_CPDAG, componets, unfinished_nodes)

    while len(mapping_res):
        componet_now = mapping_res.pop()
        true_nodes_now = componet_now[0]
        X_data_now = componet_now[2] 
        
        if len(true_nodes_now) == 1:
            continue
        
        X_ga, X_ng = fa.extract_Gaussian_and_nonGaussian(X_data_now, true_nodes_now, s_alpha)
    
        if len(X_ng) == 0:
            continue
        
        for i in X_ga:
            for j in X_ng:
                if CPDAG[i][j] == -1 and CPDAG[j][i] == -1:
                    CPDAG[j][i] = 1
            
        if len(X_ng) == 1:
            continue

        residuals_on_Gaussian_list = fa.regress_nonGaussian_on_Gaussian(X_data_now, X_ga, X_ng)

        sub_CPDAG_now_now = utils.get_subCPDAG(CPDAG, X_ng)
        componets_now_now = utils.get_all_connected_componets(sub_CPDAG_now_now, X_ng)

        mapping_res_now_now = utils.map_CPDAG_to_subCPDAG(residuals_on_Gaussian_list, CPDAG, componets_now_now, X_ng)
        if len(mapping_res_now_now)!=1:
            mapping_res += mapping_res_now_now
            continue

        CPDAG, x_0 = fa.get_ancestor_follow_skeleton(residuals_on_Gaussian_list, CPDAG, X_ng, l_alpha, i_alpha)

    return CPDAG


def identify_direction_begin_with_CPDAG_new_3(X, DAG, s_alpha, l_alpha, i_alpha):
    X_data = X.copy()
    std = np.std(X_data, axis=0)
    std[std == 0] = 1.0
    mean = np.mean(X_data, axis=0)
    X_data = (X_data - mean) / std
    
    node_num = len(DAG)
    CPDAG = np.copy(DAG)
    
    unfinished_nodes, unfinished_edges, finished_nodes, finished_edges = utils.get_unfinished_edges_nodes(CPDAG)
    
    if len(unfinished_nodes) == 0:
        return CPDAG
    
    sub_CPDAG = utils.get_subCPDAG(CPDAG, unfinished_nodes)
    X_data = utils.extract_data_unfinished(X_data, CPDAG, unfinished_nodes, finished_nodes)
    componets = utils.get_all_connected_componets(sub_CPDAG, unfinished_nodes)
    mapping_res = utils.map_CPDAG_to_subCPDAG(X_data, sub_CPDAG, componets, unfinished_nodes)

    while len(mapping_res):
        componet_now = mapping_res.pop()
        true_nodes_now = componet_now[0]
        sub_CPDAG_now = componet_now[1]
        X_data_now = componet_now[2] 
        
        if len(true_nodes_now) == 1:
            continue
        
        X_ga, X_ng = fa.extract_Gaussian_and_nonGaussian(X_data_now, true_nodes_now, s_alpha)
    
        if len(X_ng) == 0:
            continue
        
        for i in X_ga:
            for j in X_ng:
                if CPDAG[i][j] == -1 and CPDAG[j][i] == -1:
                    CPDAG[j][i] = 1
            
        if len(X_ng) == 1:
            continue

        residuals_on_Gaussian_list = fa.regress_nonGaussian_on_Gaussian(X_data_now, X_ga, X_ng)

        sub_CPDAG_now_now = utils.get_subCPDAG(CPDAG, X_ng)
        componets_now_now = utils.get_all_connected_componets(sub_CPDAG_now_now, X_ng)

        mapping_res_now_now = utils.map_CPDAG_to_subCPDAG(residuals_on_Gaussian_list, CPDAG, componets_now_now, X_ng)
        if len(mapping_res_now_now)!=1:
            mapping_res += mapping_res_now_now
            continue
        
        sub_CPDAG_res = PL.PC_LiNGAM(residuals_on_Gaussian_list, sub_CPDAG_now_now, s_alpha)
        
        # 使用NumPy高级索引优化矩阵更新
        n = len(X_ng)
        X_ng_array = np.array(X_ng)
        idx = np.ix_(X_ng_array, X_ng_array)
        CPDAG[idx] = sub_CPDAG_res
                
    return CPDAG
