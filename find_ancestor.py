import numpy as np
import copy
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import utils

import lingam
from lingam.hsic import hsic_test_gamma

import networkx as nx
# from causallearn.search.ConstraintBased.PC import pc
import pandas as pd
import statsmodels.api as sm
import time

from hyppo.independence import Hsic
from hyppo.independence import Dcorr
from hyppo.conditional import KCI

from causallearn.utils.KCI.KCI import KCI_CInd, KCI_UInd
from causallearn.utils.cit import CIT

import warnings
warnings.filterwarnings("ignore")


def make_unfinished(data):
    n_features = data.shape[1]
    return [[i, j] for i in range(n_features) for j in range(i + 1, n_features)]


def get_ancestor_pairwise_HSIC(x_data, y_data, l_alpha, i_alpha):    
    if utils.is_linear(x_data, y_data, l_alpha):
        reg = LinearRegression(fit_intercept=False)
        
        # 计算 ri_j 和 rj_i
        x_data_reshaped = x_data.reshape(-1, 1)
        y_data_reshaped = y_data.reshape(-1, 1)
        
        reg.fit(y_data_reshaped, x_data)
        ri_j = x_data - reg.predict(y_data_reshaped)
        
        reg.fit(x_data_reshaped, y_data)
        rj_i = y_data - reg.predict(x_data_reshaped)
        
        # HSIC测试
        # _, pi_j = Hsic().test(ri_j, y_data, auto=True)
        # _, pj_i = Hsic().test(rj_i, x_data, auto=True)

        _, pi_j = hsic_test_gamma(ri_j, y_data)
        _, pj_i = hsic_test_gamma(rj_i, x_data)
        
        # print("pi_j: ", pi_j, " pj_i: ", pj_i)
        
        # 判断因果关系
        if pi_j > i_alpha and pj_i <= i_alpha:
            return 1  # y_data -> x_data (j是i的祖先)
        elif pi_j <= i_alpha and pj_i > i_alpha:
            return 2  # x_data -> y_data (i是j的祖先)
        elif pi_j > i_alpha and pj_i > i_alpha:
            return 3  # 两边都独立
        elif pi_j <= i_alpha and pj_i <= i_alpha:
            return 4  # 有BCA
    else:
        return 0
    

def get_ancestor_pairwise_KCI(x_data, y_data, l_alpha, i_alpha):    
    # if utils.is_linear(x_data, y_data, l_alpha):
    if True:
        reg = LinearRegression(fit_intercept=False)
        
        # 计算 ri_j 和 rj_i
        x_data_reshaped = x_data.reshape(-1, 1)
        y_data_reshaped = y_data.reshape(-1, 1)
        
        reg.fit(y_data_reshaped, x_data)
        ri_j = x_data - reg.predict(y_data_reshaped)
        
        reg.fit(x_data_reshaped, y_data)
        rj_i = y_data - reg.predict(x_data_reshaped)
        
        # KCI测试
        # _, pi_j = Hsic().test(ri_j, y_data, auto=True)
        # _, pj_i = Hsic().test(rj_i, x_data, auto=True)
        kci_object = KCI_UInd()
        pi_j, _ = kci_object.compute_pvalue(ri_j.reshape(-1, 1), y_data.reshape(-1, 1))
        pj_i, _ = kci_object.compute_pvalue(rj_i.reshape(-1, 1), x_data.reshape(-1, 1))
        
        # print("pi_j: ", pi_j, " pj_i: ", pj_i)
        
        # 判断因果关系
        if pi_j > i_alpha and pj_i <= i_alpha:
            return 1  # y_data -> x_data (j是i的祖先)
        elif pi_j <= i_alpha and pj_i > i_alpha:
            return 2  # x_data -> y_data (i是j的祖先)
        elif pi_j > i_alpha and pj_i > i_alpha:
            return 3  # 两边都独立
        elif pi_j <= i_alpha and pj_i <= i_alpha:
            return 4  # 有BCA
    else:
        return 0
    

def get_ancestor_pairwise_KCI_final(x_data, y_data, l_alpha, i_alpha):    
    # if utils.is_linear(x_data, y_data, l_alpha):
    if True:
        reg = LinearRegression(fit_intercept=False)
        
        # 计算 ri_j 和 rj_i
        x_data_reshaped = x_data.reshape(-1, 1)
        y_data_reshaped = y_data.reshape(-1, 1)
        
        reg.fit(y_data_reshaped, x_data)
        ri_j = x_data - reg.predict(y_data_reshaped)
        
        reg.fit(x_data_reshaped, y_data)
        rj_i = y_data - reg.predict(x_data_reshaped)
        
        # KCI测试
        # _, pi_j = Hsic().test(ri_j, y_data, auto=True)
        # _, pj_i = Hsic().test(rj_i, x_data, auto=True)
        kci_object = KCI_UInd()
        pi_j, _ = kci_object.compute_pvalue(ri_j.reshape(-1, 1), y_data.reshape(-1, 1))
        pj_i, _ = kci_object.compute_pvalue(rj_i.reshape(-1, 1), x_data.reshape(-1, 1))
        
        # print("pi_j: ", pi_j, " pj_i: ", pj_i)
        
        # 判断因果关系
        if pi_j > pj_i:
            return 1  # y_data -> x_data (j是i的祖先)
        elif pi_j <= pj_i:
            return 2  # x_data -> y_data (i是j的祖先)
        else:
            pass
        # elif pi_j > i_alpha and pj_i > i_alpha:
        #     return 3  # 两边都独立
        # elif pi_j <= i_alpha and pj_i <= i_alpha:
        #     return 4  # 有BCA
    else:
        return 0


def get_CA_data(X, CA):
    return X[:, CA]


# def get_ancestor_follow_skeleton(X, CPDAG, X_ng, l_alpha, i_alpha):
#     node_num = X.shape[1]
#     # print("node number: ", node_num)
#     unfinished_edge = list()
#     map_dict = {X_ng[i]: i for i in range(len(X_ng))}
    
#     if len(X_ng) == 1:
#         return CPDAG, X_ng[0]
    
#     ancestor_dict = dict()
#     for i in range(node_num):
#         ancestor_dict[i] = []
    
    
#     ancestor_dict = {i: [] for i in range(node_num)}
#     for i in range(node_num):
#         start = X_ng[i]
#         for j in range(i + 1, node_num):
#             end = X_ng[j]
#             if CPDAG[end][start]:
#                 unfinished_edge.append([i, j])
#     # print(unfinished_edge) 
    
#     res = -1
#     x_0 = -1
#     flag_anc = True
#     flag_unf = True
#     record_pair = list()
#     # while flag_nextloop:
#     while flag_anc and flag_unf:
#         # flag_nextloop = False
#         flag_anc = False
#         flag_unf = False
#         unfinished_list_temp = copy.deepcopy(unfinished_edge)
#         for c in unfinished_list_temp:
#             i = c[0] # 映射
#             j = c[1] # 映射
#             start = X_ng[i] # 真
#             end = X_ng[j] # 真
            

#             record_pair = c

#             if CPDAG[start][end] == 1:
#                 unfinished_edge.remove(c)
#                 flag_anc = True
#                 flag_unf = True
#                 res = 1
#                 ancestor_dict[i].append(j)
#                 if x_0 == start or x_0 == -1:
#                         x_0 = end
#             elif CPDAG[end][start] == 1:
#                 unfinished_edge.remove(c)
#                 flag_anc = True
#                 flag_unf = True
#                 res = 2
#                 ancestor_dict[j].append(i)
#                 if x_0 == end or x_0 == -1:
#                         x_0 = start
#             else:
#                 x_data = X[:, i]
#                 y_data = X[:, j]

#                 CA = list(set(ancestor_dict[i]).intersection(set(ancestor_dict[j]))) # 交集

#                 # print(start, " and ", end)
#                 if len(CA) != 0:
#                     flag_unf = True
#                     # print("skip")
#                     # print("have BCA")
#                     # print("x0 = ", x_0)
#                     continue
#                 else:
#                     res = get_ancestor_pairwise_HSIC(x_data, y_data, l_alpha, i_alpha)
#                 ################################
                
                
#                 ################################
#                     if res == 0: 
#                         unfinished_edge.remove(c)
#                         flag_anc = True
#                         flag_unf = True
#                     elif res == 1:
#                         unfinished_edge.remove(c)
#                         flag_anc = True
#                         flag_unf = True
#                         CPDAG[start][end] = 1
#                         ancestor_dict[i].append(j)
#                         if x_0 == start or x_0 == -1:
#                             x_0 = end
#                     elif res == 2:
#                         unfinished_edge.remove(c)
#                         flag_anc = True
#                         flag_unf = True
#                         CPDAG[end][start] = 1
#                         ancestor_dict[j].append(i)
#                         if x_0 == end or x_0 == -1:
#                             x_0 = start
#                     elif res == 3:
#                         unfinished_edge.remove(c)
#                         flag_anc = True
#                         flag_unf = True
#                     else:
#                         flag_unf = True
#             # print("res = ", res)
#             # print("x0 = ", x_0)
#             # print("==============")
#     if x_0 == -1:
#         if len(unfinished_edge) == 0:
#             unfinished_edge.append(record_pair)
#         for c in unfinished_edge:
#             i = c[0]
#             j = c[1]
#             start = X_ng[i] # 真
#             end = X_ng[j] # 真
#             x_data = X[:, i]
#             y_data = X[:, j]

#             x_data_z = x_data
#             y_data_z = y_data

#             # x_data_z = (x_data - np.mean(x_data)) / np.std(x_data)
#             # y_data_z = (y_data - np.mean(y_data)) / np.std(y_data)
            
#             reg = LinearRegression(fit_intercept=False)
#             res = reg.fit(y_data_z.reshape(-1, 1), x_data_z) # x, y
#             coef = res.coef_
#             ri_j = x_data_z - coef * y_data_z # ri_j = xi - alpha * xj, x_data是目标变量
            
#             res = reg.fit(x_data_z.reshape(-1, 1), y_data_z) # x, y 
#             coef = res.coef_
#             rj_i = y_data_z - coef * x_data_z # rj_i = xj - alpha * xi
                                
#             _, pi_j = Hsic().test(ri_j, y_data_z, auto=True)
#             _, pj_i = Hsic().test(rj_i, x_data_z, auto=True)

#             if pi_j > pj_i:
#                 CPDAG[start][end] = 1
#                 ancestor_dict[i].append(j)
#                 if x_0 == start or x_0 == -1:
#                     x_0 = end
#             else:
#                 CPDAG[end][start] = 1
#                 ancestor_dict[j].append(i)
#                 if x_0 == end or x_0 == -1:
#                     x_0 = start
#             # print("sad, the new x0 is ", x_0)
#     for i in range(len(CPDAG[0])):
#         if CPDAG[i][x_0] == -1 and CPDAG[x_0][i] == -1:
#             CPDAG[i][x_0] = 1
#     return CPDAG, x_0


def get_ancestor_follow_skeleton(X, CPDAG, X_ng, l_alpha, i_alpha):
    sample_size = X.shape[0]
    node_num = X.shape[1]
    X_sub = []
    if sample_size >= 1500:
        indices = np.random.choice(sample_size, 1500, replace=False)
        X_sub = X[indices, :]
    else:
        X_sub = np.copy(X)

    # print(X_sub.shape)
    unfinished_edge = []
    # map_dict = {X_ng[i]: i for i in range(len(X_ng))}

    if len(X_ng) == 1:
        return CPDAG, X_ng[0]

    ancestor_dict = {i: set() for i in range(node_num)}

    # 生成unfinished_edge
    for i in range(node_num):
        start = X_ng[i]
        for j in range(i + 1, node_num):
            end = X_ng[j]
            if CPDAG[end][start]:
                unfinished_edge.append((i, j))

    # for i in range()
    
    x_0 = -1
    record_pair = []
    while True:
        flag_anc, flag_unf = False, False
        for i, j in unfinished_edge[:]:
            start, end = X_ng[i], X_ng[j]
            record_pair = [i, j]
            if CPDAG[start][end] == 1:
                unfinished_edge.remove((i, j))
                flag_anc = flag_unf = True
                ancestor_dict[i].add(j)
                if x_0 == -1 or x_0 == start:
                    x_0 = end
            elif CPDAG[end][start] == 1:
                unfinished_edge.remove((i, j))
                flag_anc = flag_unf = True
                ancestor_dict[j].add(i)
                if x_0 == -1 or x_0 == end:
                    x_0 = start
            else:
                x_data, y_data = X_sub[:, i], X_sub[:, j]
                CA = ancestor_dict[i].intersection(ancestor_dict[j])
                if len(CA) == 0:
                    res = get_ancestor_pairwise_HSIC(x_data, y_data, l_alpha, i_alpha)
                    if res == 1:
                        unfinished_edge.remove((i, j))
                        CPDAG[start][end] = 1
                        ancestor_dict[i].add(j)
                        flag_anc = flag_unf = True
                        if x_0 == -1 or x_0 == start:
                            x_0 = end
                    elif res == 2:
                        unfinished_edge.remove((i, j))
                        CPDAG[end][start] = 1
                        ancestor_dict[j].add(i)
                        flag_anc = flag_unf = True
                        if x_0 == -1 or x_0 == end:
                            x_0 = start
                    elif res == 3:
                        unfinished_edge.remove((i, j))
                        flag_anc = flag_unf = True

        if not flag_anc and not flag_unf:
            break

    if x_0 == -1:
        model = lingam.DirectLiNGAM()
        model.fit(X_sub)
        x_0 = X_ng[model._causal_order[0]]
    print(x_0)
    # if x_0 == -1:
    #     if len(unfinished_edge) == 0:
    #         unfinished_edge.append(record_pair)
    #     i, j = unfinished_edge[0]
    #     start, end = X_ng[i], X_ng[j]
    #     x_data, y_data = X[:, i], X[:, j]

    #     x_data_reshaped = x_data.reshape(-1, 1)
    #     y_data_reshaped = y_data.reshape(-1, 1)
    #     reg = LinearRegression(fit_intercept=False)
    #     reg.fit(y_data_reshaped, x_data)
    #     ri_j = x_data - reg.predict(y_data_reshaped)

    #     reg.fit(x_data_reshaped, y_data)
    #     rj_i = y_data - reg.predict(x_data_reshaped)

    #     _, pi_j = Hsic().test(ri_j, y_data, auto=True)
    #     _, pj_i = Hsic().test(rj_i, x_data, auto=True)

    #     if pi_j > pj_i:
    #         CPDAG[start][end] = 1
    #         ancestor_dict[i].add(j)
    #         x_0 = end if x_0 == -1 else x_0
    #     else:
    #         CPDAG[end][start] = 1
    #         ancestor_dict[j].add(i)
    #         x_0 = start if x_0 == -1 else x_0

    # if x_0 != -1:
    for i in range(len(CPDAG)):
        if CPDAG[i][x_0] == -1 and CPDAG[x_0][i] == -1:
            CPDAG[i][x_0] = 1

    return CPDAG, x_0


def get_ancestor_follow_skeleton_2(X, CPDAG, X_ng, l_alpha, i_alpha):
    sample_size = X.shape[0]
    node_num = X.shape[1]
    X_sub = []
    unfinished_edge = []
    # map_dict = {X_ng[i]: i for i in range(len(X_ng))}
    if sample_size >= 1500:
        indices = np.random.choice(sample_size, 1500, replace=False)
        X_sub = X[indices, :]
    else:
        X_sub = np.copy(X)

    if len(X_ng) == 1:
        return CPDAG, X_ng[0]
    
    # 生成unfinished_edge
    for i in range(node_num):
        start = X_ng[i]
        for j in range(i + 1, node_num):
            end = X_ng[j]
            if CPDAG[end][start]:
                unfinished_edge.append((i, j))

    ancestor_dict = {i: set() for i in range(node_num)}

    while True:
        flag_anc, flag_unf = False, False
        unfinished_edge_temp = np.copy(unfinished_edge)
        for i, j in unfinished_edge_temp[:]:
            start, end = X_ng[i], X_ng[j]
            record_pair = [i, j]
            x_data, y_data = X_sub[:, i], X_sub[:, j]
            CA = ancestor_dict[i].intersection(ancestor_dict[j])
            if len(CA) == 0:
                res = get_ancestor_pairwise_KCI(x_data, y_data, l_alpha, i_alpha)
                # if res == 0:
                #     unfinished_edge.remove((i, j))
                #     flag_unf = True
                if res == 1:
                    unfinished_edge.remove((i, j))
                    CPDAG[start][end] = 1
                    ancestor_dict[i].add(j)
                    flag_anc = flag_unf = True
                    # if x_0 == -1 or x_0 == start:
                    #     x_0 = end
                elif res == 2:
                    unfinished_edge.remove((i, j))
                    CPDAG[end][start] = 1
                    ancestor_dict[j].add(i)
                    flag_anc = flag_unf = True
                    # if x_0 == -1 or x_0 == end:
                    #     x_0 = start
                elif res == 3:
                    unfinished_edge.remove((i, j))
                    flag_anc = flag_unf = True
                else:
                    continue
            else:
                reg = LinearRegression(fit_intercept=False)
                x_data_reshaped = x_data.reshape(-1, 1)
                y_data_reshaped = y_data.reshape(-1, 1)
                print(CA)
                CA_list = np.array(CA, dtype=int)
                CA_data = X[:, CA_list]

                reg.fit(CA_data, x_data_reshaped)
                x_data_on_CA = x_data - reg.predict(CA_data)

                reg.fit(CA_data, y_data_reshaped)
                y_data_on_CA = y_data - reg.predict(CA_data)

                print(CA_data.shape, x_data_on_CA.type, y_data_on_CA.shape)
                res = get_ancestor_pairwise_KCI(x_data_on_CA , y_data_on_CA, l_alpha, i_alpha)
                # if res == 0:
                #     unfinished_edge.remove((i, j))
                #     flag_unf = True
                if res == 1:
                    unfinished_edge.remove((i, j))
                    CPDAG[start][end] = 1
                    ancestor_dict[i].add(j)
                    flag_anc = flag_unf = True
                    # if x_0 == -1 or x_0 == start:
                    #     x_0 = end
                elif res == 2:
                    unfinished_edge.remove((i, j))
                    CPDAG[end][start] = 1
                    ancestor_dict[j].add(i)
                    flag_anc = flag_unf = True
                    # if x_0 == -1 or x_0 == end:
                    #     x_0 = start
                elif res == 3:
                    unfinished_edge.remove((i, j))
                    flag_anc = flag_unf = True
                else:
                    continue
                # KCI_CInd.compute_value()
        if not flag_anc and not flag_unf:
            break

    for i, j in unfinished_edge[:]:
        start, end = X_ng[i], X_ng[j]
        record_pair = [i, j]
        x_data, y_data = X_sub[:, i], X_sub[:, j]
        CA = ancestor_dict[i].intersection(ancestor_dict[j])
        if len(CA) == 0:
            res = get_ancestor_pairwise_KCI_final(x_data, y_data, l_alpha, i_alpha)
            if res == 1:
                # unfinished_edge.remove((i, j))
                CPDAG[start][end] = 1
                # ancestor_dict[i].add(j)
                # flag_anc = flag_unf = True
                # if x_0 == -1 or x_0 == start:
                #     x_0 = end
            elif res == 2:
                # unfinished_edge.remove((i, j))
                CPDAG[end][start] = 1
                # ancestor_dict[j].add(i)
                # flag_anc = flag_unf = True
        else:
            reg = LinearRegression(fit_intercept=False)
            x_data_reshaped = x_data.reshape(-1, 1)
            y_data_reshaped = y_data.reshape(-1, 1)
            print(CA)
            CA_list = np.array(CA, dtype=int)
            CA_data = X[:, CA_list]

            reg.fit(CA_data, x_data_reshaped)
            x_data_on_CA = x_data - reg.predict(CA_data)

            reg.fit(CA_data, y_data_reshaped)
            y_data_on_CA = y_data - reg.predict(CA_data)

            res = get_ancestor_pairwise_KCI_final(x_data_on_CA , y_data_on_CA, l_alpha, i_alpha)
            if res == 1:
                # unfinished_edge.remove((i, j))
                CPDAG[start][end] = 1
            elif res == 2:
                # unfinished_edge.remove((i, j))
                CPDAG[end][start] = 1




    # rcd = lingam.RCD(max_explanatory_num=2, cor_alpha=l_alpha, ind_alpha=i_alpha, shapiro_alpha=1)
    # rcd.fit(X_sub)
    # res_m = rcd.adjacency_matrix_
    # res_m = np.nan_to_num(res_m, nan=0.0)
    # root_list = list()
    # for i in range(len(X_ng)):
    #     sum_temp = 0
    #     for j in range(len(X_ng)):
    #         # print(i, "and ", j, "and ", res_m[i][j])
    #         # print(i, "and ", j, "and ", res_m[i][j] == np.nan)
    #         # if res_m[i][j] == np.nan:
    #         #     res_m[i][j] = 0.0
    #         sum_temp += res_m[i][j]
    #     if sum_temp == 0.0:
    #         root_list.append(X_ng[i])
        
    # for i in range(len(X_ng)):
    #     for j in range(len(X_ng)):
    #         if res_m[i][j]:
    #             CPDAG[X_ng[i]][X_ng[j]] = 1.0
    #             CPDAG[X_ng[j]][X_ng[i]] = -1.0

    # # print(X_sub.shape)
    # print(res_m)
    # print(root_list)
    return CPDAG, 0
    

def get_ancestor_follow_skeleton_3(X, CPDAG, X_ng, l_alpha, i_alpha):
    sample_size = X.shape[0]
    node_num = X.shape[1]
    X_sub = []
    unfinished_edge = []
    # map_dict = {X_ng[i]: i for i in range(len(X_ng))}
    if sample_size >= 1500:
        indices = np.random.choice(sample_size, 1500, replace=False)
        X_sub = X[indices, :]
    else:
        X_sub = np.copy(X)

    if len(X_ng) == 1:
        return CPDAG, X_ng[0]
    
    # 生成unfinished_edge
    for i in range(node_num):
        start = X_ng[i]
        for j in range(i + 1, node_num):
            end = X_ng[j]
            if CPDAG[end][start]:
                unfinished_edge.append((i, j))

    ancestor_dict = {i: set() for i in range(node_num)}

    while True:
        flag_anc, flag_unf = False, False
        unfinished_edge_temp = np.copy(unfinished_edge)
        for i, j in unfinished_edge_temp[:]:
            start, end = X_ng[i], X_ng[j]
            record_pair = [i, j]
            x_data, y_data = X_sub[:, i], X_sub[:, j]
            CA = ancestor_dict[i].intersection(ancestor_dict[j])
            if len(CA) == 0:
                res = get_ancestor_pairwise_KCI(x_data, y_data, l_alpha, i_alpha)
                # if res == 0:
                #     unfinished_edge.remove((i, j))
                #     flag_unf = True
                if res == 1:
                    unfinished_edge.remove((i, j))
                    CPDAG[start][end] = 1
                    ancestor_dict[i].add(j)
                    flag_anc = flag_unf = True
                    if x_0 == -1 or x_0 == start:
                        x_0 = end
                elif res == 2:
                    unfinished_edge.remove((i, j))
                    CPDAG[end][start] = 1
                    ancestor_dict[j].add(i)
                    flag_anc = flag_unf = True
                    if x_0 == -1 or x_0 == end:
                        x_0 = start
                elif res == 3:
                    unfinished_edge.remove((i, j))
                    flag_anc = flag_unf = True
                else:
                    continue
            else:
                reg = LinearRegression(fit_intercept=False)
                x_data_reshaped = x_data.reshape(-1, 1)
                y_data_reshaped = y_data.reshape(-1, 1)
                CA_data = X[:, CA]

                reg.fit(CA_data, x_data_reshaped)
                x_data_on_CA = x_data - reg.predict(CA_data)

                reg.fit(CA_data, y_data_reshaped)
                y_data_on_CA = y_data - reg.predict(CA_data)

                res = get_ancestor_pairwise_KCI(x_data_on_CA , y_data_on_CA, l_alpha, i_alpha)
                # if res == 0:
                #     unfinished_edge.remove((i, j))
                #     flag_unf = True
                if res == 1:
                    unfinished_edge.remove((i, j))
                    CPDAG[start][end] = 1
                    ancestor_dict[i].add(j)
                    flag_anc = flag_unf = True
                    if x_0 == -1 or x_0 == start:
                        x_0 = end
                elif res == 2:
                    unfinished_edge.remove((i, j))
                    CPDAG[end][start] = 1
                    ancestor_dict[j].add(i)
                    flag_anc = flag_unf = True
                    if x_0 == -1 or x_0 == end:
                        x_0 = start
                elif res == 3:
                    unfinished_edge.remove((i, j))
                    flag_anc = flag_unf = True
                else:
                    continue
                # KCI_CInd.compute_value()
        if not flag_anc and not flag_unf:
            break

    
    # rcd = lingam.RCD(max_explanatory_num=2, cor_alpha=l_alpha, ind_alpha=i_alpha, shapiro_alpha=1)
    # rcd.fit(X_sub)
    # res_m = rcd.adjacency_matrix_
    # res_m = np.nan_to_num(res_m, nan=0.0)
    # root_list = list()
    # for i in range(len(X_ng)):
    #     sum_temp = 0
    #     for j in range(len(X_ng)):
    #         # print(i, "and ", j, "and ", res_m[i][j])
    #         # print(i, "and ", j, "and ", res_m[i][j] == np.nan)
    #         # if res_m[i][j] == np.nan:
    #         #     res_m[i][j] = 0.0
    #         sum_temp += res_m[i][j]
    #     if sum_temp == 0.0:
    #         root_list.append(X_ng[i])
        
    # for i in range(len(X_ng)):
    #     for j in range(len(X_ng)):
    #         if res_m[i][j]:
    #             CPDAG[X_ng[i]][X_ng[j]] = 1.0
    #             CPDAG[X_ng[j]][X_ng[i]] = -1.0

    # # print(X_sub.shape)
    # print(res_m)
    # print(root_list)
    return CPDAG, 0


def get_ancestor_follow_skeleton_4(X, CPDAG, X_ng, l_alpha, i_alpha):
    sample_size = X.shape[0]
    node_num = X.shape[1]
    X_sub = []
    if sample_size >= 1500:
        indices = np.random.choice(sample_size, 1500, replace=False)
        X_sub = X[indices, :]
    else:
        X_sub = np.copy(X)

    # print(X_sub.shape)
    unfinished_edge = []
    # map_dict = {X_ng[i]: i for i in range(len(X_ng))}

    if len(X_ng) == 1:
        return CPDAG, X_ng[0]

    ancestor_dict = {i: set() for i in range(node_num)}

    # 生成unfinished_edge
    for i in range(node_num):
        start = X_ng[i]
        for j in range(i + 1, node_num):
            end = X_ng[j]
            if CPDAG[end][start]:
                unfinished_edge.append((i, j))

    # for i in range()
    
    x_0 = -1
    record_pair = []
    while True:
        flag_anc, flag_unf = False, False
        for i, j in unfinished_edge[:]:
            start, end = X_ng[i], X_ng[j]
            record_pair = [i, j]
            if CPDAG[start][end] == 1:
                unfinished_edge.remove((i, j))
                flag_anc = flag_unf = True
                ancestor_dict[i].add(j)
                if x_0 == -1 or x_0 == start:
                    x_0 = end
            elif CPDAG[end][start] == 1:
                unfinished_edge.remove((i, j))
                flag_anc = flag_unf = True
                ancestor_dict[j].add(i)
                if x_0 == -1 or x_0 == end:
                    x_0 = start
            else:
                x_data, y_data = X_sub[:, i], X_sub[:, j]
                CA = ancestor_dict[i].intersection(ancestor_dict[j])
                if len(CA) == 0:
                    res = get_ancestor_pairwise_HSIC(x_data, y_data, l_alpha, i_alpha)
                    if res == 1:
                        unfinished_edge.remove((i, j))
                        CPDAG[start][end] = 1
                        ancestor_dict[i].add(j)
                        flag_anc = flag_unf = True
                        if x_0 == -1 or x_0 == start:
                            x_0 = end
                    elif res == 2:
                        unfinished_edge.remove((i, j))
                        CPDAG[end][start] = 1
                        ancestor_dict[j].add(i)
                        flag_anc = flag_unf = True
                        if x_0 == -1 or x_0 == end:
                            x_0 = start
                    elif res == 3:
                        unfinished_edge.remove((i, j))
                        flag_anc = flag_unf = True

        if not flag_anc and not flag_unf:
            break

    if x_0 == -1:
        model = lingam.DirectLiNGAM()
        model.fit(X_sub)
        x_0 = X_ng[model._causal_order[0]]
    print(x_0)
    # if x_0 == -1:
    #     if len(unfinished_edge) == 0:
    #         unfinished_edge.append(record_pair)
    #     i, j = unfinished_edge[0]
    #     start, end = X_ng[i], X_ng[j]
    #     x_data, y_data = X[:, i], X[:, j]

    #     x_data_reshaped = x_data.reshape(-1, 1)
    #     y_data_reshaped = y_data.reshape(-1, 1)
    #     reg = LinearRegression(fit_intercept=False)
    #     reg.fit(y_data_reshaped, x_data)
    #     ri_j = x_data - reg.predict(y_data_reshaped)

    #     reg.fit(x_data_reshaped, y_data)
    #     rj_i = y_data - reg.predict(x_data_reshaped)

    #     _, pi_j = Hsic().test(ri_j, y_data, auto=True)
    #     _, pj_i = Hsic().test(rj_i, x_data, auto=True)

    #     if pi_j > pj_i:
    #         CPDAG[start][end] = 1
    #         ancestor_dict[i].add(j)
    #         x_0 = end if x_0 == -1 else x_0
    #     else:
    #         CPDAG[end][start] = 1
    #         ancestor_dict[j].add(i)
    #         x_0 = start if x_0 == -1 else x_0

    # if x_0 != -1:
    for i in range(len(CPDAG)):
        if CPDAG[i][x_0] == -1 and CPDAG[x_0][i] == -1:
            CPDAG[i][x_0] = 1

    return CPDAG, x_0


def extract_Gaussian_and_nonGaussian(data, true_node_list, s_alpha):
    is_gaussian = utils.is_Gaussian
    gaussian_list = []
    nonGaussian_list = []

    for i, node in enumerate(true_node_list):
        if is_gaussian(data[:, i], s_alpha):
            gaussian_list.append(node)
        else:
            nonGaussian_list.append(node)
    
    return np.array(gaussian_list, dtype=np.int32), np.array(nonGaussian_list, dtype=np.int32)


# def regress_nonGaussian_on_Gaussian(data, gaussian_list, nonGaussian_list):
#     len_gaussian = gaussian_list.shape[0]
#     if len_gaussian == 0:
#         return data

#     # 合并并排序索引列表
#     combined_indices = np.argsort(np.hstack((gaussian_list, nonGaussian_list)))
#     # print(combined_indices)
#     # combined_list = sorted(np.hstack((gaussian_list, nonGaussian_list)))
#     # # print(combined_list)
#     # 提取排序后的数据
#     sorted_data = data[:, combined_indices]
    
#     # 分割高斯和非高斯数据
#     # len_gaussian = len(gaussian_list)
#     data_Gaussian = sorted_data[:, :len_gaussian]
#     data_nonGaussian = sorted_data[:, len_gaussian:]
    
#     # 批量回归和预测
#     reg = LinearRegression(fit_intercept=False)
#     reg.fit(data_Gaussian, data_nonGaussian)
#     predictions = reg.predict(data_Gaussian)
    
#     # 计算残差
#     residual_nonGaussian = data_nonGaussian - predictions

#     return residual_nonGaussian


def regress_nonGaussian_on_Gaussian(data, gaussian_list, nonGaussian_list):
    if gaussian_list.shape[0] == 0:
        return data
    
    data_Gaussian = []
    data_nonGaussian = []
    
    # # print(type(gaussian_list))
    total_list = sorted(np.hstack((gaussian_list, nonGaussian_list)))
    # # print(total_list)
    for i, node in enumerate(total_list):
        # # print(i, " and ", node)
        if node in gaussian_list:
            data_Gaussian.append(data[:, i])
        else:
            data_nonGaussian.append(data[:, i])

    data_Gaussian = np.array(data_Gaussian).T

    data_nonGaussian = np.array(data_nonGaussian).T
    
    # 批量回归和预测
    reg = LinearRegression(fit_intercept=False)
    # reg = Lasso(alpha=0.1)
    reg.fit(data_Gaussian, data_nonGaussian)
    predictions = reg.predict(data_Gaussian)
    
    # 计算残差
    residual_nonGaussian = data_nonGaussian - predictions

    return residual_nonGaussian


# x_0与nonGaussian_list里都是真实节点的index
def regress_nonGaussian_on_x0(data, x_0, nonGaussian_list):
    # 找到 x_0_data 和 data_nonGaussian
    x_0_index = np.argwhere(nonGaussian_list == x_0)
    x_0_data = data[:, x_0_index].reshape(-1, 1)
    nonGaussian_indices = [i for i, node in enumerate(nonGaussian_list) if node != x_0]
    data_nonGaussian = data[:, nonGaussian_indices]

    reg = LinearRegression(fit_intercept=False)
    # reg = Lasso(alpha=0.1)
    
    # 对所有非高斯变量进行回归并计算残差
    reg.fit(x_0_data, data_nonGaussian)
    predictions = reg.predict(x_0_data)
    residual_nonGaussian = data_nonGaussian - predictions

    return residual_nonGaussian


# def regress_undirected_on_directed(data, und_node, di_node, ancestor_dict):


