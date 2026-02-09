import numpy as np
import copy
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import utils

import lingam
from lingam.hsic import hsic_test_gamma

import networkx as nx
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

# 模块级别的对象重用
_KCI_OBJECT = KCI_UInd()
_REG_OBJECT = LinearRegression(fit_intercept=False)


def make_unfinished(data):
    n_features = data.shape[1]
    return [[i, j] for i in range(n_features) for j in range(i + 1, n_features)]


def _get_sampled_data(X, max_samples=1500):
    """统一的采样函数"""
    if X.shape[0] <= max_samples:
        return X.copy()
    indices = np.random.choice(X.shape[0], max_samples, replace=False)
    return X[indices, :]


def get_ancestor_pairwise_HSIC(x_data, y_data, l_alpha, i_alpha):    
    if utils.is_linear(x_data, y_data, l_alpha):
        # 使用全局回归对象
        reg = _REG_OBJECT
        
        # 计算 ri_j 和 rj_i
        x_data_reshaped = x_data[:, np.newaxis]
        y_data_reshaped = y_data[:, np.newaxis]
        
        reg.fit(y_data_reshaped, x_data)
        ri_j = x_data - np.dot(y_data_reshaped, reg.coef_.T).ravel()
        
        reg.fit(x_data_reshaped, y_data)
        rj_i = y_data - np.dot(x_data_reshaped, reg.coef_.T).ravel()
        
        # HSIC测试
        _, pi_j = hsic_test_gamma(ri_j, y_data)
        _, pj_i = hsic_test_gamma(rj_i, x_data)
        
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
    

def get_ancestor_pairwise_KCI(x_data, y_data, l_alpha, i_alpha, kci_obj=_KCI_OBJECT):    
    if True:
        reg = _REG_OBJECT
        
        # 计算 ri_j 和 rj_i
        x_data_reshaped = x_data[:, np.newaxis]
        y_data_reshaped = y_data[:, np.newaxis]
        
        reg.fit(y_data_reshaped, x_data)
        ri_j = x_data - np.dot(y_data_reshaped, reg.coef_.T).ravel()
        
        reg.fit(x_data_reshaped, y_data)
        rj_i = y_data - np.dot(x_data_reshaped, reg.coef_.T).ravel()
        
        # KCI测试
        pi_j, _ = kci_obj.compute_pvalue(ri_j[:, np.newaxis], y_data[:, np.newaxis])
        pj_i, _ = kci_obj.compute_pvalue(rj_i[:, np.newaxis], x_data[:, np.newaxis])
        
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
    

def get_ancestor_pairwise_KCI_final(x_data, y_data, l_alpha, i_alpha, kci_obj=_KCI_OBJECT):    
    if True:
        reg = _REG_OBJECT
        
        # 计算 ri_j 和 rj_i
        x_data_reshaped = x_data[:, np.newaxis]
        y_data_reshaped = y_data[:, np.newaxis]
        
        reg.fit(y_data_reshaped, x_data)
        ri_j = x_data - np.dot(y_data_reshaped, reg.coef_.T).ravel()
        
        reg.fit(x_data_reshaped, y_data)
        rj_i = y_data - np.dot(x_data_reshaped, reg.coef_.T).ravel()
        
        # KCI测试
        pi_j, _ = kci_obj.compute_pvalue(ri_j[:, np.newaxis], y_data[:, np.newaxis])
        pj_i, _ = kci_obj.compute_pvalue(rj_i[:, np.newaxis], x_data[:, np.newaxis])
        
        # 判断因果关系
        if pi_j > pj_i:
            return 1  # y_data -> x_data (j是i的祖先)
        elif pi_j <= pj_i:
            return 2  # x_data -> y_data (i是j的祖先)
        else:
            pass
    else:
        return 0


def get_CA_data(X, CA):
    return X[:, CA]


def get_ancestor_follow_skeleton(X, CPDAG, X_ng, l_alpha, i_alpha):
    X_sub = _get_sampled_data(X)
    node_num = X.shape[1]

    if len(X_ng) == 1:
        return CPDAG, X_ng[0]

    ancestor_dict = {i: set() for i in range(node_num)}

    # 生成unfinished_edge
    unfinished_edge = set()
    for i in range(node_num):
        start = X_ng[i]
        for j in range(i + 1, node_num):
            end = X_ng[j]
            if CPDAG[end][start]:
                unfinished_edge.add((i, j))
    
    x_0 = -1
    record_pair = []
    
    while True:
        flag_anc, flag_unf = False, False
        to_remove = []
        
        for i, j in unfinished_edge:
            start, end = X_ng[i], X_ng[j]
            record_pair = [i, j]
            
            if CPDAG[start][end] == 1:
                to_remove.append((i, j))
                flag_anc = flag_unf = True
                ancestor_dict[i].add(j)
                if x_0 == -1:
                    x_0 = end
                elif x_0 == start:
                    x_0 = end
            elif CPDAG[end][start] == 1:
                to_remove.append((i, j))
                flag_anc = flag_unf = True
                ancestor_dict[j].add(i)
                if x_0 == -1:
                    x_0 = start
                elif x_0 == end:
                    x_0 = start
            else:
                x_data, y_data = X_sub[:, i], X_sub[:, j]
                CA = ancestor_dict[i].intersection(ancestor_dict[j])
                if len(CA) == 0:
                    res = get_ancestor_pairwise_HSIC(x_data, y_data, l_alpha, i_alpha)
                    if res == 1:
                        to_remove.append((i, j))
                        CPDAG[start][end] = 1
                        ancestor_dict[i].add(j)
                        flag_anc = flag_unf = True
                        if x_0 == -1:
                            x_0 = end
                        elif x_0 == start:
                            x_0 = end
                    elif res == 2:
                        to_remove.append((i, j))
                        CPDAG[end][start] = 1
                        ancestor_dict[j].add(i)
                        flag_anc = flag_unf = True
                        if x_0 == -1:
                            x_0 = start
                        elif x_0 == end:
                            x_0 = start
                    elif res == 3:
                        to_remove.append((i, j))
                        flag_anc = flag_unf = True
        
        # 批量删除
        for item in to_remove:
            unfinished_edge.discard(item)
        
        if not flag_anc and not flag_unf:
            break

    if x_0 == -1:
        model = lingam.DirectLiNGAM()
        model.fit(X_sub)
        x_0 = X_ng[model._causal_order[0]]

    for i in range(len(CPDAG)):
        if CPDAG[i][x_0] == -1 and CPDAG[x_0][i] == -1:
            CPDAG[i][x_0] = 1

    return CPDAG, x_0


def get_ancestor_follow_skeleton_2(X, CPDAG, X_ng, l_alpha, i_alpha):
    X_sub = _get_sampled_data(X)
    node_num = X.shape[1]

    if len(X_ng) == 1:
        return CPDAG, X_ng[0]
    
    # 生成unfinished_edge
    unfinished_edge = set()
    for i in range(node_num):
        start = X_ng[i]
        for j in range(i + 1, node_num):
            end = X_ng[j]
            if CPDAG[end][start]:
                unfinished_edge.add((i, j))

    ancestor_dict = {i: set() for i in range(node_num)}
    reg = _REG_OBJECT

    while True:
        flag_anc, flag_unf = False, False
        to_remove = []
        
        for i, j in list(unfinished_edge):
            start, end = X_ng[i], X_ng[j]
            record_pair = [i, j]
            x_data, y_data = X_sub[:, i], X_sub[:, j]
            CA = ancestor_dict[i].intersection(ancestor_dict[j])
            
            if len(CA) == 0:
                res = get_ancestor_pairwise_KCI(x_data, y_data, l_alpha, i_alpha)
                if res == 1:
                    to_remove.append((i, j))
                    CPDAG[start][end] = 1
                    ancestor_dict[i].add(j)
                    flag_anc = flag_unf = True
                elif res == 2:
                    to_remove.append((i, j))
                    CPDAG[end][start] = 1
                    ancestor_dict[j].add(i)
                    flag_anc = flag_unf = True
                elif res == 3:
                    to_remove.append((i, j))
                    flag_anc = flag_unf = True
            else:
                x_data_reshaped = x_data[:, np.newaxis]
                y_data_reshaped = y_data[:, np.newaxis]
                CA_list = np.array(list(CA), dtype=int)
                CA_data = X[:, CA_list]

                reg.fit(CA_data, x_data_reshaped)
                x_data_on_CA = x_data - reg.predict(CA_data).ravel()

                reg.fit(CA_data, y_data_reshaped)
                y_data_on_CA = y_data - reg.predict(CA_data).ravel()

                res = get_ancestor_pairwise_KCI(x_data_on_CA, y_data_on_CA, l_alpha, i_alpha)
                if res == 1:
                    to_remove.append((i, j))
                    CPDAG[start][end] = 1
                    ancestor_dict[i].add(j)
                    flag_anc = flag_unf = True
                elif res == 2:
                    to_remove.append((i, j))
                    CPDAG[end][start] = 1
                    ancestor_dict[j].add(i)
                    flag_anc = flag_unf = True
                elif res == 3:
                    to_remove.append((i, j))
                    flag_anc = flag_unf = True
        
        # 批量删除
        for item in to_remove:
            unfinished_edge.discard(item)
        
        if not flag_anc and not flag_unf:
            break

    # 处理剩余的边
    for i, j in list(unfinished_edge):
        start, end = X_ng[i], X_ng[j]
        x_data, y_data = X_sub[:, i], X_sub[:, j]
        CA = ancestor_dict[i].intersection(ancestor_dict[j])
        
        if len(CA) == 0:
            res = get_ancestor_pairwise_KCI_final(x_data, y_data, l_alpha, i_alpha)
            if res == 1:
                CPDAG[start][end] = 1
            elif res == 2:
                CPDAG[end][start] = 1
        else:
            x_data_reshaped = x_data[:, np.newaxis]
            y_data_reshaped = y_data[:, np.newaxis]
            CA_list = np.array(list(CA), dtype=int)
            CA_data = X[:, CA_list]

            reg.fit(CA_data, x_data_reshaped)
            x_data_on_CA = x_data - reg.predict(CA_data).ravel()

            reg.fit(CA_data, y_data_reshaped)
            y_data_on_CA = y_data - reg.predict(CA_data).ravel()

            res = get_ancestor_pairwise_KCI_final(x_data_on_CA, y_data_on_CA, l_alpha, i_alpha)
            if res == 1:
                CPDAG[start][end] = 1
            elif res == 2:
                CPDAG[end][start] = 1

    return CPDAG, 0


def get_ancestor_follow_skeleton_3(X, CPDAG, X_ng, l_alpha, i_alpha):
    X_sub = _get_sampled_data(X)
    node_num = X.shape[1]

    if len(X_ng) == 1:
        return CPDAG, X_ng[0]
    
    # 生成unfinished_edge
    unfinished_edge = set()
    for i in range(node_num):
        start = X_ng[i]
        for j in range(i + 1, node_num):
            end = X_ng[j]
            if CPDAG[end][start]:
                unfinished_edge.add((i, j))

    ancestor_dict = {i: set() for i in range(node_num)}
    reg = _REG_OBJECT
    x_0 = -1

    while True:
        flag_anc, flag_unf = False, False
        to_remove = []
        
        for i, j in list(unfinished_edge):
            start, end = X_ng[i], X_ng[j]
            x_data, y_data = X_sub[:, i], X_sub[:, j]
            CA = ancestor_dict[i].intersection(ancestor_dict[j])
            
            if len(CA) == 0:
                res = get_ancestor_pairwise_KCI(x_data, y_data, l_alpha, i_alpha)
                if res == 1:
                    to_remove.append((i, j))
                    CPDAG[start][end] = 1
                    ancestor_dict[i].add(j)
                    flag_anc = flag_unf = True
                    if x_0 == -1:
                        x_0 = end
                    elif x_0 == start:
                        x_0 = end
                elif res == 2:
                    to_remove.append((i, j))
                    CPDAG[end][start] = 1
                    ancestor_dict[j].add(i)
                    flag_anc = flag_unf = True
                    if x_0 == -1:
                        x_0 = start
                    elif x_0 == end:
                        x_0 = start
                elif res == 3:
                    to_remove.append((i, j))
                    flag_anc = flag_unf = True
            else:
                x_data_reshaped = x_data[:, np.newaxis]
                y_data_reshaped = y_data[:, np.newaxis]
                CA_list = np.array(list(CA), dtype=int)
                CA_data = X[:, CA_list]

                reg.fit(CA_data, x_data_reshaped)
                x_data_on_CA = x_data - reg.predict(CA_data).ravel()

                reg.fit(CA_data, y_data_reshaped)
                y_data_on_CA = y_data - reg.predict(CA_data).ravel()

                res = get_ancestor_pairwise_KCI(x_data_on_CA, y_data_on_CA, l_alpha, i_alpha)
                if res == 1:
                    to_remove.append((i, j))
                    CPDAG[start][end] = 1
                    ancestor_dict[i].add(j)
                    flag_anc = flag_unf = True
                    if x_0 == -1:
                        x_0 = end
                    elif x_0 == start:
                        x_0 = end
                elif res == 2:
                    to_remove.append((i, j))
                    CPDAG[end][start] = 1
                    ancestor_dict[j].add(i)
                    flag_anc = flag_unf = True
                    if x_0 == -1:
                        x_0 = start
                    elif x_0 == end:
                        x_0 = start
                elif res == 3:
                    to_remove.append((i, j))
                    flag_anc = flag_unf = True
        
        # 批量删除
        for item in to_remove:
            unfinished_edge.discard(item)
        
        if not flag_anc and not flag_unf:
            break

    return CPDAG, 0


def get_ancestor_follow_skeleton_4(X, CPDAG, X_ng, l_alpha, i_alpha):
    X_sub = _get_sampled_data(X)
    node_num = X.shape[1]

    if len(X_ng) == 1:
        return CPDAG, X_ng[0]

    ancestor_dict = {i: set() for i in range(node_num)}

    # 生成unfinished_edge
    unfinished_edge = set()
    for i in range(node_num):
        start = X_ng[i]
        for j in range(i + 1, node_num):
            end = X_ng[j]
            if CPDAG[end][start]:
                unfinished_edge.add((i, j))
    
    x_0 = -1
    
    while True:
        flag_anc, flag_unf = False, False
        to_remove = []
        
        for i, j in unfinished_edge:
            start, end = X_ng[i], X_ng[j]
            
            if CPDAG[start][end] == 1:
                to_remove.append((i, j))
                flag_anc = flag_unf = True
                ancestor_dict[i].add(j)
                if x_0 == -1:
                    x_0 = end
                elif x_0 == start:
                    x_0 = end
            elif CPDAG[end][start] == 1:
                to_remove.append((i, j))
                flag_anc = flag_unf = True
                ancestor_dict[j].add(i)
                if x_0 == -1:
                    x_0 = start
                elif x_0 == end:
                    x_0 = start
            else:
                x_data, y_data = X_sub[:, i], X_sub[:, j]
                CA = ancestor_dict[i].intersection(ancestor_dict[j])
                if len(CA) == 0:
                    res = get_ancestor_pairwise_HSIC(x_data, y_data, l_alpha, i_alpha)
                    if res == 1:
                        to_remove.append((i, j))
                        CPDAG[start][end] = 1
                        ancestor_dict[i].add(j)
                        flag_anc = flag_unf = True
                        if x_0 == -1:
                            x_0 = end
                        elif x_0 == start:
                            x_0 = end
                    elif res == 2:
                        to_remove.append((i, j))
                        CPDAG[end][start] = 1
                        ancestor_dict[j].add(i)
                        flag_anc = flag_unf = True
                        if x_0 == -1:
                            x_0 = start
                        elif x_0 == end:
                            x_0 = start
                    elif res == 3:
                        to_remove.append((i, j))
                        flag_anc = flag_unf = True
        
        # 批量删除
        for item in to_remove:
            unfinished_edge.discard(item)
        
        if not flag_anc and not flag_unf:
            break

    if x_0 == -1:
        model = lingam.DirectLiNGAM()
        model.fit(X_sub)
        x_0 = X_ng[model._causal_order[0]]

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


def regress_nonGaussian_on_Gaussian(data, gaussian_list, nonGaussian_list):
    if gaussian_list.shape[0] == 0:
        return data
    
    # 使用布尔索引优化
    total_list = np.hstack((gaussian_list, nonGaussian_list))
    sort_idx = np.argsort(total_list)
    sorted_list = total_list[sort_idx]
    
    # 创建布尔掩码
    gaussian_mask = np.isin(sorted_list, gaussian_list)
    
    # 提取数据
    sorted_data = data[:, sort_idx]
    data_Gaussian = sorted_data[:, gaussian_mask]
    data_nonGaussian = sorted_data[:, ~gaussian_mask]
    
    # 批量回归和预测
    reg = _REG_OBJECT
    reg.fit(data_Gaussian, data_nonGaussian)
    predictions = reg.predict(data_Gaussian)
    
    # 计算残差
    residual_nonGaussian = data_nonGaussian - predictions

    return residual_nonGaussian


def regress_nonGaussian_on_x0(data, x_0, nonGaussian_list):
    # 使用布尔索引优化
    mask = nonGaussian_list == x_0
    x_0_index = np.where(mask)[0][0]
    x_0_data = data[:, x_0_index:x_0_index+1]
    
    # 获取其他非高斯变量
    data_nonGaussian = data[:, ~mask]

    reg = _REG_OBJECT
    
    # 对所有非高斯变量进行回归并计算残差
    reg.fit(x_0_data, data_nonGaussian)
    predictions = reg.predict(x_0_data)
    residual_nonGaussian = data_nonGaussian - predictions

    return residual_nonGaussian
