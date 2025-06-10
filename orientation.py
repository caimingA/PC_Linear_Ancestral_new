import numpy as np
import find_ancestor as fa
import utils
# import find_skeleton as fa
import copy
from multiprocessing import Pool
import time

import PC_LiNGAM as PL


# def identify_direction(X, CPDAG, collider_dict, collider_midnode_list):
def identify_direction_begin_with_CPDAG(X, DAG, s_alpha, l_alpha, i_alpha):
    X_data = copy.deepcopy(X)
    node_num = len(DAG)
    node_list = [i for i in range(node_num)]
    # ancestor_dict = dict()
    # for i in node_list:
    #     ancestor_dict[i] = []
    # line 1
    CPDAG, colliders = utils.get_CPDAG(DAG)
    collider_dict, collider_midnode_list = fa.extract_colliders(colliders)
    direct_nodes, des_res = utils.get_descendants_of_all_colliders(CPDAG, collider_midnode_list)
    
    # line 2
    E_di = direct_nodes
    E_ud = list(set(node_list) - set(direct_nodes))
    # # print("directed edges:　", E_di)
    # # print("undirected edges:　"E_ud)

    # line 3 - 4
    X_ga = list()
    X_ng = list()

    X_ga, X_ng = fa.extract_Gaussian_and_nonGaussian(X_data, s_alpha)
    # for i in E_ud:
    #     if utils.is_Gaussian(X[:, i], s_alpha):
    #         X_ga.append(i)
    #     else:
    #         X_ng.append(i) 

    # line 5 - 7
    if len(X_ng) == 0:
        return CPDAG
    
    while len(X_ng) != 0:
        # line 9
        for i in X_ga:
            for j in X_ng:
                if CPDAG[i][j] == -1 and CPDAG[j][i] == -1:
                    CPDAG[j][i] = 1
        # line 10
        residuals_on_Gaussian_list = fa.regress_nonGaussian_on_Gaussian(X_data, X_ga, X_ng)
        # line 11-12
        CPDAG, x_0 = fa.get_ancestor_follow_skeleton(residuals_on_Gaussian_list, CPDAG, X_ng, l_alpha, i_alpha)
        if len(X_ng) == 1 and x_0 == X_ng[0]:
            X_ng = list()
        else:
            # index = np.where(X_ng==x_0)
            # X_ng = np.delete(X_ng, np.where(X_ng==x_0))

            # line 13
            residuals_on_x0_list = fa.regress_nonGaussian_on_x0(residuals_on_Gaussian_list, x_0, X_ng)
            X_ng = np.delete(X_ng, np.where(X_ng==x_0))
            
            X_ga_temp, X_ng_temp = fa.extract_Gaussian_and_nonGaussian(residuals_on_x0_list, s_alpha)
            X_data = residuals_on_x0_list
            X_ga_next = list()
            X_ng_next = list()
            for i in X_ga_temp:
                X_ga_next.append(X_ng[i])
            for i in X_ng_temp:
                X_ng_next.append(X_ng[i])
            X_ng = X_ng_next
            X_ga = X_ga_next
    return CPDAG


def identify_direction_begin_with_CPDAG_revised(X, DAG, s_alpha, l_alpha, i_alpha):
    X_data = copy.deepcopy(X)
    node_num = len(DAG)
    # node_list = [i for i in range(node_num)]
    
    # line 1
    CPDAG, colliders = utils.get_CPDAG(DAG)
    
    unfinished_nodes, unfinished_edges, finished_nodes, finished_edges = utils.get_unfinished_edges_nodes(CPDAG)
    
    # line 2
    
    # 进行了重新编号
    sub_CPDAG = utils.get_subCPDAG(CPDAG, unfinished_nodes) 
    # 通过编号对数据进行了重新排序
    X_data = utils.extract_data_unfinished(X_data, CPDAG, unfinished_nodes, finished_nodes)

    # 获取重新编号后的componet
    componets = utils.get_all_connected_componets(sub_CPDAG)

    node_list_temp = [i for i in range(node_num)]
    # 得到所有的组分的真实节点编号，子CPDAG和对应的数据
    mapping_res = utils.map_CPDAG_to_subCPDAG(X_data, sub_CPDAG, componets, unfinished_nodes)
    
    # # print("directed edges:　", E_di)
    # # print("undirected edges:　"E_ud)
    while len(mapping_res):
        # # print(len(mapping_res))
        componet_now = mapping_res.pop()
        true_nodes_now = componet_now[0]
        sub_CPDAG_now = componet_now[1]
        X_data_now = componet_now[2] 
        # line 3 - 4
        X_ga = list()
        X_ng = list()

        # 得到的是虚拟编号（sub_CPDAG_now）
        X_ga, X_ng = fa.extract_Gaussian_and_nonGaussian(X_data_now, s_alpha)

    # line 5 - 7
        if len(X_ng) == 0:
            # return CPDAG #######
            continue
        
        while len(X_ng) != 0:
            # line 9
            for i in X_ga:
                for j in X_ng:
                    if sub_CPDAG_now[i][j] == -1 and sub_CPDAG_now[j][i] == -1:
                        sub_CPDAG_now[j][i] = 1
            # line 10
            residuals_on_Gaussian_list = fa.regress_nonGaussian_on_Gaussian(X_data_now, X_ga, X_ng)
            
            sub_CPDAG_now_now = utils.get_subCPDAG(sub_CPDAG, X_ng)
            componets_now_now = utils.get_all_connected_componets(sub_CPDAG_now_now)
            mapping_res_now_now = utils.map_CPDAG_to_subCPDAG(residuals_on_Gaussian_list, sub_CPDAG_now_now, componets_now_now, true_nodes_now)
            if len(mapping_res_now_now)!=1:
                mapping_res += mapping_res_now_now
                break
            # line 11-12
            sub_CPDAG_now, x_0 = fa.get_ancestor_follow_skeleton(residuals_on_Gaussian_list, sub_CPDAG_now, X_ng, l_alpha, i_alpha)
            
            if len(X_ng) == 1 and x_0 == X_ng[0]:
                X_ng = list()
            else:
                # index = np.where(X_ng==x_0)
                # X_ng = np.delete(X_ng, np.where(X_ng==x_0))

                # line 13
                residuals_on_x0_list = fa.regress_nonGaussian_on_x0(residuals_on_Gaussian_list, x_0, X_ng)
                # true_nodes_now = np.delete(true_nodes_now, np.where(true_nodes_now==true_nodes_now[x_0])) # 保持数量一致
                X_ng = np.delete(X_ng, np.where(X_ng==x_0)) # 保持数量一致
                sub_CPDAG_now_now = utils.get_subCPDAG(sub_CPDAG, X_ng)
                componets_now_now = utils.get_all_connected_componets(sub_CPDAG_now_now)
                mapping_res_now_now = utils.map_CPDAG_to_subCPDAG(residuals_on_x0_list, sub_CPDAG_now_now, componets_now_now, true_nodes_now)
                if len(mapping_res_now_now)!=1:
                    mapping_res += mapping_res_now_now
                    break
                X_ga_temp, X_ng_temp = fa.extract_Gaussian_and_nonGaussian(residuals_on_x0_list, s_alpha)
                X_data = residuals_on_x0_list
                X_ga_next = list()
                X_ng_next = list()
                for i in X_ga_temp:
                    X_ga_next.append(X_ng[i])
                for i in X_ng_temp:
                    X_ng_next.append(X_ng[i])
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
    X_data = copy.deepcopy(X)
    mean = np.mean(X_data, axis=0)  # 对每一列（特征）计算均值
    std = np.std(X_data, axis=0)    # 对每一列（特征）计算标准差
    node_num = len(DAG)
    X_data = (X_data - mean) / std
    # node_list = [i for i in range(node_num)]
    
    # line 1
    # 将已经还有边为定向的点与已经定向的点区分开
    # CPDAG, colliders = utils.get_CPDAG(DAG)
    CPDAG = np.copy(DAG)
    
    unfinished_nodes, unfinished_edges, finished_nodes, finished_edges = utils.get_unfinished_edges_nodes(CPDAG)
    # # print(ancestor_dict)
    
    if len(unfinished_nodes) == 0:
        return CPDAG
    # 进行了重新编号
    sub_CPDAG = utils.get_subCPDAG(CPDAG, unfinished_nodes)

    # 通过编号对数据进行了重新排序
    X_data = utils.extract_data_unfinished(X_data, CPDAG, unfinished_nodes, finished_nodes)

    # 获取重新编号后的componet
    componets = utils.get_all_connected_componets(sub_CPDAG, unfinished_nodes)

    # 得到所有的组分的真实节点编号，子CPDAG和对应的数据
    mapping_res = utils.map_CPDAG_to_subCPDAG(X_data, sub_CPDAG, componets, unfinished_nodes)

    while len(mapping_res):
        componet_now = mapping_res.pop()
        true_nodes_now = componet_now[0]
        # sub_CPDAG_now = componet_now[1]
        X_data_now = componet_now[2] 
        
        # # print("now nodes: ", true_nodes_now)
        if len(true_nodes_now) == 1:
            continue
        
        # line 3 - 4
        X_ga = list()
        X_ng = list()

        # 
        X_ga, X_ng = fa.extract_Gaussian_and_nonGaussian(X_data_now, true_nodes_now, s_alpha)
    
        # print("1, ", X_data_now.shape[0])
        if len(X_ng) == 0:
            continue
        
        while len(X_ng) != 0:
            for i in X_ga:
                for j in X_ng:
                    if CPDAG[i][j] == -1 and CPDAG[j][i] == -1:
                        CPDAG[j][i] = 1 # i -> j
            
            if len(X_ng) == 1:
                X_ng = list()
                continue

            # line 10
            residuals_on_Gaussian_list = fa.regress_nonGaussian_on_Gaussian(X_data_now, X_ga, X_ng)

            sub_CPDAG_now_now = utils.get_subCPDAG(CPDAG, X_ng)
            componets_now_now = utils.get_all_connected_componets(sub_CPDAG_now_now, X_ng)

            mapping_res_now_now = utils.map_CPDAG_to_subCPDAG(residuals_on_Gaussian_list, CPDAG, componets_now_now, X_ng)
            if len(mapping_res_now_now)!=1:
                mapping_res += mapping_res_now_now
                break

            # line 11-12
            CPDAG, x_0 = fa.get_ancestor_follow_skeleton(residuals_on_Gaussian_list, CPDAG, X_ng, l_alpha, i_alpha)
            
            if len(X_ng) == 1:
                X_ng = list()
            else:
                residuals_on_x0_list = fa.regress_nonGaussian_on_x0(residuals_on_Gaussian_list, x_0, X_ng)
                X_ng = np.delete(X_ng, np.where(X_ng==x_0)) # 保持数量一致
                # print("4, ", residuals_on_x0_list.shape[0])
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
    X_data = copy.deepcopy(X)
    mean = np.mean(X_data, axis=0)  # 对每一列（特征）计算均值
    std = np.std(X_data, axis=0)    # 对每一列（特征）计算标准差
    node_num = len(DAG)
    X_data = (X_data - mean) / std
    # node_list = [i for i in range(node_num)]
    
    # line 1
    # 将已经还有边为定向的点与已经定向的点区分开
    # CPDAG, colliders = utils.get_CPDAG(DAG)
    CPDAG = np.copy(DAG)
    
    unfinished_nodes, unfinished_edges, finished_nodes, finished_edges = utils.get_unfinished_edges_nodes(CPDAG)
    # # print(ancestor_dict)
    
    if len(unfinished_nodes) == 0:
        return CPDAG
    # 进行了重新编号
    sub_CPDAG = utils.get_subCPDAG(CPDAG, unfinished_nodes)

    # 通过编号对数据进行了重新排序
    X_data = utils.extract_data_unfinished(X_data, CPDAG, unfinished_nodes, finished_nodes)

    # 获取重新编号后的componet
    componets = utils.get_all_connected_componets(sub_CPDAG, unfinished_nodes)

    # 得到所有的组分的真实节点编号，子CPDAG和对应的数据
    mapping_res = utils.map_CPDAG_to_subCPDAG(X_data, sub_CPDAG, componets, unfinished_nodes)

    while len(mapping_res):
        componet_now = mapping_res.pop()
        true_nodes_now = componet_now[0]
        # sub_CPDAG_now = componet_now[1]
        X_data_now = componet_now[2] 
        
        # # print("now nodes: ", true_nodes_now)
        if len(true_nodes_now) == 1:
            continue
        
        # line 3 - 4
        X_ga = list()
        X_ng = list()

        # 
        X_ga, X_ng = fa.extract_Gaussian_and_nonGaussian(X_data_now, true_nodes_now, s_alpha)
    
        # print("1, ", X_data_now.shape[0])
        if len(X_ng) == 0:
            # return CPDAG #######
            continue
        # if len(X_ng) == 1:
        #     X_ng = list()
        #     continue
        
        # while len(X_ng) != 0:
            # line 9
            # print("X_ng = ", X_ng)
            # print("X_ga = ", X_ga)
        for i in X_ga:
            for j in X_ng:
                if CPDAG[i][j] == -1 and CPDAG[j][i] == -1:
                    CPDAG[j][i] = 1 # i -> j
            
        if len(X_ng) == 1:
            X_ng = list()
            continue

            # line 10
        residuals_on_Gaussian_list = fa.regress_nonGaussian_on_Gaussian(X_data_now, X_ga, X_ng)
            # print("2, ", residuals_on_Gaussian_list.shape[0])

        sub_CPDAG_now_now = utils.get_subCPDAG(CPDAG, X_ng)
        componets_now_now = utils.get_all_connected_componets(sub_CPDAG_now_now, X_ng)

        mapping_res_now_now = utils.map_CPDAG_to_subCPDAG(residuals_on_Gaussian_list, CPDAG, componets_now_now, X_ng)
        if len(mapping_res_now_now)!=1:
            mapping_res += mapping_res_now_now
            break

            # line 11-12
            # # print("size = ", len(residuals_on_Gaussian_list[0]), " and ", len(X_ng))
        # CPDAG, x_0 = fa.get_ancestor_follow_skeleton_2(residuals_on_Gaussian_list, CPDAG, X_ng, l_alpha, i_alpha)
        CPDAG, x_0 = fa.get_ancestor_follow_skeleton(residuals_on_Gaussian_list, CPDAG, X_ng, l_alpha, i_alpha)
            # print("3, ", residuals_on_Gaussian_list.shape[0])
            # if x_0 == -1:
            #     break
    return CPDAG


def identify_direction_begin_with_CPDAG_new_3(X, DAG, s_alpha, l_alpha, i_alpha):
    X_data = copy.deepcopy(X)
    mean = np.mean(X_data, axis=0)  # 对每一列（特征）计算均值
    std = np.std(X_data, axis=0)    # 对每一列（特征）计算标准差
    node_num = len(DAG)
    X_data = (X_data - mean) / std
    # node_list = [i for i in range(node_num)]
    
    # line 1
    # 将已经还有边为定向的点与已经定向的点区分开
    # CPDAG, colliders = utils.get_CPDAG(DAG)
    CPDAG = np.copy(DAG)
    
    unfinished_nodes, unfinished_edges, finished_nodes, finished_edges = utils.get_unfinished_edges_nodes(CPDAG)
    # # print(ancestor_dict)
    
    if len(unfinished_nodes) == 0:
        return CPDAG
    # 进行了重新编号
    sub_CPDAG = utils.get_subCPDAG(CPDAG, unfinished_nodes)

    # 通过编号对数据进行了重新排序
    X_data = utils.extract_data_unfinished(X_data, CPDAG, unfinished_nodes, finished_nodes)

    # 获取重新编号后的componet
    componets = utils.get_all_connected_componets(sub_CPDAG, unfinished_nodes)

    # 得到所有的组分的真实节点编号，子CPDAG和对应的数据
    mapping_res = utils.map_CPDAG_to_subCPDAG(X_data, sub_CPDAG, componets, unfinished_nodes)

    while len(mapping_res):
        componet_now = mapping_res.pop()
        true_nodes_now = componet_now[0]
        sub_CPDAG_now = componet_now[1]
        X_data_now = componet_now[2] 
        
        # # print("now nodes: ", true_nodes_now)
        if len(true_nodes_now) == 1:
            continue
        
        X_ga = list()
        X_ng = list()

        X_ga, X_ng = fa.extract_Gaussian_and_nonGaussian(X_data_now, true_nodes_now, s_alpha)
    
        # print("1, ", X_data_now.shape[0])
        if len(X_ng) == 0:
            # return CPDAG #######
            continue
        
        for i in X_ga:
            for j in X_ng:
                if CPDAG[i][j] == -1 and CPDAG[j][i] == -1:
                    CPDAG[j][i] = 1 # i -> j
            
        if len(X_ng) == 1:
            X_ng = list()
            continue

            # line 10
        residuals_on_Gaussian_list = fa.regress_nonGaussian_on_Gaussian(X_data_now, X_ga, X_ng)
            # print("2, ", residuals_on_Gaussian_list.shape[0])

        sub_CPDAG_now_now = utils.get_subCPDAG(CPDAG, X_ng)
        componets_now_now = utils.get_all_connected_componets(sub_CPDAG_now_now, X_ng)

        mapping_res_now_now = utils.map_CPDAG_to_subCPDAG(residuals_on_Gaussian_list, CPDAG, componets_now_now, X_ng)
        if len(mapping_res_now_now)!=1:
            mapping_res += mapping_res_now_now
            break
        
        sub_CPDAG_res = PL.PC_LiNGAM(residuals_on_Gaussian_list, sub_CPDAG_now_now, s_alpha)
        for i in range(len(X_ng)):
            for j in range(i+1, len(X_ng)):
                CPDAG[X_ng[i]][X_ng[j]] = sub_CPDAG_res[i][j]
                CPDAG[X_ng[j]][X_ng[i]] = sub_CPDAG_res[j][i]
        
        # sub_CPDAG_res = PL.PC_LiNGAM(X_data_now, sub_CPDAG_now, s_alpha)
        # for i in range(len(true_nodes_now)):
        #     for j in range(i+1, len(true_nodes_now)):
        #         CPDAG[true_nodes_now[i]][true_nodes_now[j]] = sub_CPDAG_res[i][j]
        #         CPDAG[true_nodes_now[j]][true_nodes_now[i]] = sub_CPDAG_res[j][i]
        # # line 3 - 4
        # X_ga = list()
        # X_ng = list()

        # # # 
        # X_ga, X_ng = fa.extract_Gaussian_and_nonGaussian(X_data_now, true_nodes_now, s_alpha)
    
        # # # print("1, ", X_data_now.shape[0])
        # if len(X_ng) == 0:
        #     # return CPDAG #######
        #     continue
        # # # if len(X_ng) == 1:
        # # #     X_ng = list()
        # # #     continue
        # X_ng_map = [i for i in range(len(true_nodes_now)) if true_nodes_now[i] in X_ng]
        # X_data_nonGaussian = X_data_now[:, X_ng_map]
        # # while len(X_ng) != 0:
        #     # line 9
        #     # print("X_ng = ", X_ng)
        #     # print("X_ga = ", X_ga)
        # for i in X_ga:
        #     for j in X_ng:
        #         if CPDAG[i][j] == -1 and CPDAG[j][i] == -1:
        #             CPDAG[j][i] = 1 # i -> j
            
        # if len(X_ng) == 1:
        #     X_ng = list()
        #     continue

        #     # line 10
        # residuals_on_Gaussian_list = fa.regress_nonGaussian_on_Gaussian(X_data_now, X_ga, X_ng)
        #     # print("2, ", residuals_on_Gaussian_list.shape[0])

        # sub_CPDAG_now_now = utils.get_subCPDAG(CPDAG, X_ng)
        # componets_now_now = utils.get_all_connected_componets(sub_CPDAG_now_now, X_ng)

        # mapping_res_now_now = utils.map_CPDAG_to_subCPDAG(residuals_on_Gaussian_list, CPDAG, componets_now_now, X_ng)
        # if len(mapping_res_now_now)!=1:
        #     mapping_res += mapping_res_now_now
        #     break

        #     # line 11-12
        #     # # print("size = ", len(residuals_on_Gaussian_list[0]), " and ", len(X_ng))
        # CPDAG, x_0 = fa.get_ancestor_follow_skeleton_2(residuals_on_Gaussian_list, CPDAG, X_ng, l_alpha, i_alpha)
        #     # print("3, ", residuals_on_Gaussian_list.shape[0])
        #     # if x_0 == -1:
        #     #     break
    return CPDAG

# X, DAG, s_alpha, l_alpha, i_alpha
def identify_direction_begin_with_CPDAG_multithread(experiment_set):
    X = experiment_set[0]
    DAG = experiment_set[1]
    s_alpha = experiment_set[2]
    l_alpha = experiment_set[3]
    i_alpha = experiment_set[4]

    X_data = copy.deepcopy(X)
    node_num = len(DAG)
    # node_list = [i for i in range(node_num)]
    start_time = time.process_time()
    # line 1
    # 将已经还有边为定向的点与已经定向的点区分开
    CPDAG, colliders = utils.get_CPDAG(DAG)
    true_CPDAG = copy.deepcopy(CPDAG)
    unfinished_nodes, unfinished_edges, finished_nodes, finished_edges = utils.get_unfinished_edges_nodes(CPDAG)
    # # print(ancestor_dict)
    
    if len(unfinished_nodes) == 0:
        return CPDAG
    # 进行了重新编号
    sub_CPDAG = utils.get_subCPDAG(CPDAG, unfinished_nodes)

    # 通过编号对数据进行了重新排序
    X_data = utils.extract_data_unfinished(X_data, CPDAG, unfinished_nodes, finished_nodes)

    # 获取重新编号后的componet
    componets = utils.get_all_connected_componets(sub_CPDAG, unfinished_nodes)

    # 得到所有的组分的真实节点编号，子CPDAG和对应的数据
    mapping_res = utils.map_CPDAG_to_subCPDAG(X_data, sub_CPDAG, componets, unfinished_nodes)

    while len(mapping_res):
        componet_now = mapping_res.pop()
        true_nodes_now = componet_now[0]
        # sub_CPDAG_now = componet_now[1]
        X_data_now = componet_now[2] 
        
        # # print("now nodes: ", true_nodes_now)
        if len(true_nodes_now) == 1:
            continue
        
        # line 3 - 4
        X_ga = list()
        X_ng = list()

        # 
        X_ga, X_ng = fa.extract_Gaussian_and_nonGaussian(X_data_now, true_nodes_now, s_alpha)
    
        # print("1, ", X_data_now.shape[0])
        if len(X_ng) == 0:
            # return CPDAG #######
            continue
        # if len(X_ng) == 1:
        #     X_ng = list()
        #     continue
        
        while len(X_ng) != 0:
            # line 9
            # print("X_ng = ", X_ng)
            # print("X_ga = ", X_ga)
            for i in X_ga:
                for j in X_ng:
                    if CPDAG[i][j] == -1 and CPDAG[j][i] == -1:
                        CPDAG[j][i] = 1 # i -> j
            
            if len(X_ng) == 1:
                X_ng = list()
                continue

            # line 10
            residuals_on_Gaussian_list = fa.regress_nonGaussian_on_Gaussian(X_data_now, X_ga, X_ng)
            # print("2, ", residuals_on_Gaussian_list.shape[0])

            sub_CPDAG_now_now = utils.get_subCPDAG(CPDAG, X_ng)
            componets_now_now = utils.get_all_connected_componets(sub_CPDAG_now_now, X_ng)

            mapping_res_now_now = utils.map_CPDAG_to_subCPDAG(residuals_on_Gaussian_list, CPDAG, componets_now_now, X_ng)
            if len(mapping_res_now_now)!=1:
                mapping_res += mapping_res_now_now
                break

            # line 11-12
            # # print("size = ", len(residuals_on_Gaussian_list[0]), " and ", len(X_ng))
            CPDAG, x_0 = fa.get_ancestor_follow_skeleton(residuals_on_Gaussian_list, CPDAG, X_ng, l_alpha, i_alpha)
            # print("3, ", residuals_on_Gaussian_list.shape[0])
            # if x_0 == -1:
            #     break
            
            if len(X_ng) == 1:
                X_ng = list()
            else:
                residuals_on_x0_list = fa.regress_nonGaussian_on_x0(residuals_on_Gaussian_list, x_0, X_ng)
                X_ng = np.delete(X_ng, np.where(X_ng==x_0)) # 保持数量一致
                # print("4, ", residuals_on_x0_list.shape[0])
                sub_CPDAG_now_now = utils.get_subCPDAG(CPDAG, X_ng)
                componets_now_now = utils.get_all_connected_componets(sub_CPDAG_now_now, X_ng)

                mapping_res_now_now = utils.map_CPDAG_to_subCPDAG(residuals_on_x0_list, CPDAG, componets_now_now, X_ng)
                if len(mapping_res_now_now)!=1:
                    mapping_res += mapping_res_now_now
                    break

                X_ga, X_ng = fa.extract_Gaussian_and_nonGaussian(residuals_on_x0_list, X_ng, s_alpha)
                X_data_now = residuals_on_x0_list
                # print("4, ", X_data_now.shape[0])
                # X_ga_next = list()
                # X_ng_next = list()

    # CPDAG = utils.match_to_MEC(CPDAG, true_CPDAG, colliders)
    CPDAG = utils.do_Meek_rule(CPDAG)

    end_time = time.process_time()
    return CPDAG, end_time - start_time