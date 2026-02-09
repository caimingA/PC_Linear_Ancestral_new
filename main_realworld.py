import random
import pandas as pd
import numpy as np
import data_generator as dg
import find_ancestor as fa
import utils
import utils_v2
import find_skeleton as fs
import copy

from scipy.stats import pearsonr, shapiro
import matplotlib.pyplot as plt
import draw

import causaldag as cd
import lingam

import networkx as nx

import write_excel as we

# import orientation as ori
import orientation_v3 as ori

# import PC_LiNGAM as PL
import PC_LiNGAM_optimized as PL
from causallearn.search.ConstraintBased.PC import pc

import time

import bnlearn as bn
import xlwt

from sklearn.preprocessing import StandardScaler

# from IPython.display import Image
# from pgmpy.utils import get_example_model

# # Load the model
# asia_model = get_example_model('asia')
# print(asia_model)

# Visualize the network
# viz = asia_model.to_graphviz()
# viz.draw('asia.png', prog='neato')
# Image('asia.png')

# df = bn.import_example(data='titanic')
# print(df.head(5))

# model = bn.import_DAG('titanic')
# print(model)

# # df = bn.sampling(model, n=10)
# # print(df.head(5))

# bn.plot(model)

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid

import utils_notear


def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def write_excel(res_list, name):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet('PC',cell_overwrite_ok=True)
    sheet2 = f.add_sheet('proposed',cell_overwrite_ok=True)
    sheet3 = f.add_sheet('PL',cell_overwrite_ok=True)
    sheet4 = f.add_sheet('LiNGAM',cell_overwrite_ok=True)
    sheet5 = f.add_sheet('NoTears',cell_overwrite_ok=True)
    sheet6 = f.add_sheet('proposed true',cell_overwrite_ok=True)
    # sheet7 = f.add_sheet('PL true',cell_overwrite_ok=True)
    # sheet6 = f.add_sheet('RCD',cell_overwrite_ok=True)

    res_PC = res_list[0]
    res_proposed = res_list[1]
    res_PL = res_list[2]
    res_L = res_list[3]
    res_notears = res_list[4]
    # res_rcd = res_list[5]
    res_proposed_true = res_list[5]
    res_PL_true = res_list[6]
    
    for i in range(len(res_PC)):
        for j in range(len(res_PC[0])):
            sheet1.write(i,j, res_PC[i][j])

    for i in range(len(res_proposed)):
        for j in range(len(res_proposed[0])):
            sheet2.write(i,j, res_proposed[i][j])

    for i in range(len(res_PL)):
        for j in range(len(res_PL[0])):
            sheet3.write(i,j, res_PL[i][j])

    for i in range(len(res_L)):
        for j in range(len(res_L[0])):
            sheet4.write(i,j, res_L[i][j])

    for i in range(len(res_notears)):
        for j in range(len(res_notears[0])):
            sheet5.write(i,j, res_notears[i][j])

    for i in range(len(res_proposed_true)):
        for j in range(len(res_proposed_true[0])):
            sheet6.write(i,j, res_proposed_true[i][j])

    f.save(name + '.xls')


def change_matrix(cg_graph):
    node_num = len(cg_graph)
    matrix = np.zeros((node_num, node_num))
    for i in range(node_num):
        for j in range(node_num):
            # matrix[i][j] = cg_graph[j][i]
            matrix[i][j] = cg_graph[i][j]
    return matrix

def evaluation(true_DAG, CPDAG):
    edge_num = np.sum(true_DAG)
    CPDAG_temp = np.where(CPDAG==1.0, 1.0, 0.0)
    dis_edge_num = np.sum(CPDAG_temp)
    correct = 0
    node_num = len(true_DAG)
    for i in range(node_num):
        for j in range(node_num):
            if true_DAG[i][j] == 1 and CPDAG[i][j] == 1:
                correct += 1
            # if true_DAG[i][j] == 1 and CPDAG[i][j] == -1 and CPDAG[j][i] == -1:
            #     correct += 1
    pre = correct/dis_edge_num
    rec = correct/edge_num
    F = 0
    if pre + rec == 0:
        F = 0
    else: 
        F = 2*pre*rec / (pre + rec)
    return [pre, rec, F] 


def evaluation_CPDAG(true_CPDAG, CPDAG):
    # edge_num = np.sum(true_DAG)
    # CPDAG_temp = np.where(CPDAG==1.0, 1.0, 0.0)
    # dis_edge_num = np.sum(CPDAG_temp)
    node_num = len(true_CPDAG)
    correct = 0
    estimated_edges = 0
    true_edges = 0
    for i in range(node_num):
        for j in range(node_num):
            if i < j:
                if true_CPDAG[i][j] == 1 and true_CPDAG[j][i] == -1:
                    true_edges += 1
                elif true_CPDAG[i][j] == -1 and true_CPDAG[j][i] == 1:
                    true_edges += 1
                elif true_CPDAG[i][j] == -1 and true_CPDAG[j][i] == -1:
                    true_edges += 1
                else:
                    pass
    
    for i in range(node_num):
        for j in range(node_num):
            if i < j:
                if CPDAG[i][j] == 1 and CPDAG[j][i] == -1:
                    estimated_edges += 1
                elif CPDAG[i][j] == -1 and CPDAG[j][i] == 1:
                    estimated_edges += 1
                elif CPDAG[i][j] == -1 and CPDAG[j][i] == -1:
                    estimated_edges += 1
                else:
                    pass
    for i in range(node_num):
        for j in range(node_num):
            if i < j:
                if true_CPDAG[i][j] == 1 and true_CPDAG[j][i] == -1:
                    if CPDAG[i][j] == 1 and CPDAG[j][i] == -1:
                        correct += 1
                elif true_CPDAG[i][j] == -1 and true_CPDAG[j][i] == 1:
                    if CPDAG[i][j] == -1 and CPDAG[j][i] == 1:
                        correct += 1
                elif true_CPDAG[i][j] == -1 and true_CPDAG[j][i] == -1:
                    if CPDAG[i][j] == -1 and CPDAG[j][i] == -1:
                        correct += 1
                else:
                    pass
    # missing = 0
    # red = 0
    # node_num = len(true_CPDAG)
    # for i in range(node_num):
    #     for j in range(node_num):
    #         if true_CPDAG[i][j] == 1 and true_CPDAG[j][i] == -1:
    #             if CPDAG[i][j] == 1 and CPDAG[j][i] == -1:
    #                 correct += 1
    #             elif CPDAG[i][j] == 0 and CPDAG[j][i] == 0:
    #                 missing += 1
    #             else:
    #                 red += 1
    #                 missing += 1
    #         if true_CPDAG[i][j] == -1 and true_CPDAG[j][i] == -1:
    #             if i < j:
    #                 if true_CPDAG[i][j] == -1 and true_CPDAG[j][i] == -1:
    #                     correct += 1
    #                 elif true_CPDAG[i][j] == 0 and true_CPDAG[j][i] == 0:
    #                     missing += 1
    #                 else:
    #                     red += 1
    #                     missing += 1
    #         if true_CPDAG[i][j] == 0 and true_CPDAG[j][i] == 0:
    #             if i < j:
    #                 if true_CPDAG[i][j] == -1 and true_CPDAG[j][i] == -1:
    #                     pass
    #                 else:
    #                     red += 1

    # pre = correct/(correct + red)
    # rec = correct/(correct + missing)
    pre = correct/estimated_edges if estimated_edges != 0 else 0.0
    rec = correct/true_edges if true_edges != 0 else 0.0
    F = 0
    if pre + rec == 0:
        F = 0
    else: 
        F = 2*pre*rec / (pre + rec)
    return [pre, rec, F]

if __name__ == '__main__':
    current_time = time.localtime()
    formatted_time = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    # alpha_l_list = [0.01, 0.05, 0.1, 0.5]
    # alpha_i_list = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    # alpha_l_list = [0.01, 0.05, 0.1, 0.5]
    # alpha_i_list = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    # alpha_p = [1]

    # samples = [25, 50, 100]

    df_fmri = pd.read_excel("fMRI.xlsx", sheet_name="Sheet1", header=None)

    data_array_fmri = df_fmri.to_numpy()
    # print(data_array)
    data_array_fmri = np.where(df_fmri.to_numpy() == -1, 0, df_fmri.to_numpy())
    data_array_fmri = np.where(data_array_fmri != 0, 1.0, 0.0)
    data_array_fmri = data_array_fmri.T
    # print(data_array_fmri.T)

    # draw.draw(data_array_fmri.T)

    df_fmri_data = pd.read_excel("fMRI.xlsx", sheet_name="Sheet2", header=None)
    data_array_fmri_data = df_fmri_data.to_numpy()
    # indices = np.random.choice(data_array_fmri_data.shape[0], 100, replace=False)
    # selected_data_array_fmri_data = data_array_fmri_data[indices, : ]
    
    print(data_array_fmri_data.shape)
    # print(selected_data_array_fmri_data.shape)
    # data_array_fmri_CPDAG = np.array(
    #     [
    #         [0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
    #         [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0],
    #         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
    #         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
    #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0],
    #         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    #     ]
    # )
    data_array_fmri_CPDAG_original = utils.get_True_CPDAG(data_array_fmri, [])
    data_array_fmri_CPDAG_original.flags.writeable = False
    data_array_fmri_CPDAG_original_temp = np.where(data_array_fmri_CPDAG_original== 1.0, 1.0, 0.0)
    arcs_fmri_temp = utils.matrix_to_edge(data_array_fmri_CPDAG_original_temp)
    g_fmri_temp = cd.DAG(arcs=arcs_fmri_temp)

    collider_fmri_old = g_fmri_temp.vstructures()

    data_array_fmri_CPDAG = utils.get_True_CPDAG(data_array_fmri, [2])
    print(data_array_fmri_CPDAG)
    # input()
    
    df_sachs_data = pd.read_excel("sachs.xls", sheet_name="Sheet2")
    data_array_sachs_data = df_sachs_data.to_numpy()
    # indices = np.random.choice(data_array_sachs_data.shape[0], 100, replace=False)
    # selected_data_array_sachs_data = data_array_sachs_data[indices, : ]
    print(data_array_sachs_data)
    # print(selected_data_array_sachs_data.shape)
    
    # model = bn.import_DAG('sachs')
    # row_order = ["PKC","PKA", "Raf", "Jnk", "P38", "Mek", "Erk", "Akt", "Plcg", "PIP3", "PIP2"]
    # col_order = ["PKC","PKA", "Raf", "Jnk", "P38", "Mek", "Erk", "Akt", "Plcg", "PIP3", "PIP2"]
    # # print(model["adjmat"])
    # df_sachs = model["adjmat"]
    # df_sachs = df_sachs.reindex(index=row_order, columns=col_order)
    # print(df_sachs)
    # data_array_sachs = np.where(df_sachs.to_numpy(), 1.0, 0.0)
    # print(data_array_sachs.T)
    data_array_sachs=np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
            ]
        )
    
    # data_array_sachs_CPDAG=np.array(data_array_sachs)
    # for i in range(len(data_array_sachs_CPDAG)):
    #     for j in range(len(data_array_sachs_CPDAG)):
    #         if data_array_sachs_CPDAG[i][j] == 1:
    #             data_array_sachs_CPDAG[j][i] = -1
    data_array_sachs_CPDAG_original = utils.get_True_CPDAG(data_array_sachs, [])
    data_array_sachs_CPDAG_original.flags.writeable = False
    data_array_sachs_CPDAG_original_temp = np.where(data_array_sachs_CPDAG_original== 1.0, 1.0, 0.0)
    arcs_sachs_temp = utils.matrix_to_edge(data_array_sachs_CPDAG_original_temp)
    g_sachs_temp = cd.DAG(arcs=arcs_sachs_temp)

    collider_sachs_old = g_sachs_temp.vstructures()

    data_array_sachs_CPDAG = utils.get_True_CPDAG(data_array_sachs, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    data_array_fmri_CPDAG.flags.writeable = False
    data_array_sachs_CPDAG.flags.writeable = False
    print(data_array_sachs_CPDAG)
    # input()
    # data_array_sachs
    
    # draw.draw(data_array_sachs.T)
    # print(model["adjmat"][2:,1:], type(model["adjmat"]))

    # ex.execute_KCI_once([df_fmri, ])
    # data_array_sachs_data_copy = data_array_sachs_data
    # data_array_fmri_data_copy = data_array_fmri_data
    
    current_time = time.localtime()
    formatted_time = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    count = 0
    res_list = [[] for i in range(7)]
    res_list_sachs = [[] for i in range(7)]
    count_e_fmri = list()
    count_e_sachs = list()

    np.random.seed(2025)
    for _ in range(50):
        indices = np.random.choice(data_array_sachs_data.shape[0], 200, replace=False)
        data_array_sachs_data_temp = data_array_sachs_data[indices, : ]
        scaler = StandardScaler()
        data_array_sachs_data_temp = scaler.fit_transform(data_array_sachs_data_temp)
        indices = np.random.choice(data_array_fmri_data.shape[0], 100, replace=False)
        data_array_fmri_data_temp = data_array_fmri_data[indices, : ]
        # scaler = StandardScaler()
        # data_array_fmri_data_temp = scaler.fit_transform(data_array_fmri_data_temp)
        
        count += 1
        current_time = time.localtime()
        formatted_time = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        
        ##### fMRI
        print("=====================")
        print(data_array_fmri)
        time_5 = time.perf_counter()
        cg = pc(data_array_fmri_data_temp, 0.05, uc_priority=0, indep_test="fisherz", stable=True)
        CPDAG_PC = cg.G.graph

        DAG_temp = np.where(CPDAG_PC == 1.0, 1.0, 0.0)
        arcs = utils.matrix_to_edge(DAG_temp)
        g = cd.DAG(arcs=arcs)

        collider_old = g.vstructures()
        # print(CPDAG_PC[0][1])
        time_6 = time.perf_counter()
        time_PC = (time_6 - time_5)
        CPDAG_PC = change_matrix(CPDAG_PC)
        
        print(f"data: {data_array_fmri_data_temp[0]}")
        print(CPDAG_PC)
        print(data_array_fmri_CPDAG)
        print(evaluation_CPDAG(data_array_fmri_CPDAG, CPDAG_PC))
        print(time_PC)
        res_PC = evaluation_CPDAG(data_array_fmri_CPDAG, CPDAG_PC)
        res_PC.append(time_PC)
        res_list[0].append(res_PC)
        # draw.draw_CPDAG(CPDAG_PC)

        
        
       
        print("=====================")

        time_3 = time.perf_counter()
        # CPDAG_PL = PL.PC_LiNGAM(data_array_fmri_data_temp, CPDAG_PC, 0.05)
        CPDAG_PL = PL.PC_LiNGAM_CPDAG(data_array_fmri_data_temp, CPDAG_PC, 0.05)
        CPDAG_PL = utils.do_Meek_rule(CPDAG_PL)
        time_4 = time.perf_counter()
        time_PL = (time_4 - time_3)
        print(f"data: {data_array_fmri_data_temp[0]}")
        print(f"CPDAG_PC: \n {CPDAG_PC}")
        print(CPDAG_PL)
        print(data_array_fmri_CPDAG)
        print(evaluation_CPDAG(data_array_fmri_CPDAG, CPDAG_PL))
        print(time_PL + time_PC)
        res_PL = evaluation_CPDAG(data_array_fmri_CPDAG, CPDAG_PL)
        res_PL.append(time_PL + time_PC)
        res_list[2].append(res_PL)
        # draw.draw_CPDAG(CPDAG_PL)
        
        # time_5 = time.perf_counter()
        # cg = pc(data_array_fmri_data, indep_test="kci")
        # CPDAG_PC = cg.G.graph
        # time_6 = time.perf_counter()
        # time_PC = (time_6 - time_5)
        # print(CPDAG_PC)
        # print(time_PC)
        print("=====================")

        time_1 = time.perf_counter()
        CPDAG_proposed = ori.identify_direction_begin_with_CPDAG_new(data_array_fmri_data_temp, CPDAG_PC, 0.05, 0.01, 0.001)
        
        # print(CPDAG_proposed)
        CPDAG_proposed = utils_v2.detect_exceptions_2(CPDAG_proposed, collider_old)
        CPDAG_proposed = utils_v2.do_Meek_rule(CPDAG_proposed)
        # print("new: \n", CPDAG_proposed_new)
        # temp_flag = utils.detect_expections(CPDAG_proposed_new, collider_old)
        # if temp_flag:
        #     count_e_fmri.append(temp_flag)
        #     draw.draw_CPDAG(CPDAG_PC)
        #     draw.draw_CPDAG(CPDAG_proposed)
        #     draw.draw_CPDAG(CPDAG_proposed_new)
        # # utils.get_new_vstructure(CPDAG_proposed)
        time_2 = time.perf_counter()
        
        time_proposed = (time_2 - time_1)
        # print(f"data: {data_array_fmri_data_temp[0]}")
        # print(f"CPDAG_PC: \n {CPDAG_PC}")
        # print(CPDAG_proposed)
        # print(data_array_fmri_CPDAG)
        # print(evaluation_CPDAG(data_array_fmri_CPDAG, CPDAG_proposed))
        # print(time_proposed + time_PC)
        res_proposed = evaluation_CPDAG(data_array_fmri_CPDAG, CPDAG_proposed)
        res_proposed.append(time_proposed + time_PC)
        res_list[1].append(res_proposed)
        # draw.draw_CPDAG(CPDAG_proposed)
        print("=====================")
        
        
        time_7 = time.perf_counter()
        lingam_model = lingam.DirectLiNGAM()
        lingam_model.fit(data_array_fmri_data_temp)
        DAG_L = lingam_model.adjacency_matrix_
        DAG_L = np.where(DAG_L != 0.0, 1.0, 0.0)
        time_8 = time.perf_counter()
        time_L = (time_8 - time_7)
        # print(DAG_L)
        for i in range(len(DAG_L)):
            for j in range(len(DAG_L)):
                if DAG_L[i][j] == 1.0:
                    DAG_L[j][i] = -1.0
        
        # print(f"data: {data_array_fmri_data_temp[0]}")
        # print(DAG_L)
        # print(data_array_fmri_CPDAG)
        # print(evaluation_CPDAG(data_array_fmri_CPDAG, DAG_L))
        # print(time_L)

        res_L = evaluation_CPDAG(data_array_fmri_CPDAG, DAG_L)
        res_L.append(time_L)
        res_list[3].append(res_L)
        # draw.draw_CPDAG(DAG_L)
        print("=====================")
        time_9 = time.perf_counter()
        W_est = notears_linear(data_array_fmri_data_temp, lambda1=0.1, loss_type='l2')
        DAG_notears = np.where(W_est != 0.0, 1.0, 0.0)
        time_10 = time.perf_counter()
        time_notears = (time_10 - time_9)
        # print(DAG_L)
        for i in range(len(DAG_notears)):
            for j in range(len(DAG_notears)):
                if DAG_notears[i][j] == 1.0:
                    DAG_notears[j][i] = -1.0
        # print(f"data: {data_array_fmri_data_temp[0]}")
        # print(DAG_notears)
        # print(data_array_fmri_CPDAG)
        # print(evaluation_CPDAG(data_array_fmri_CPDAG, DAG_notears))
        # print(time_notears)

        res_notears = evaluation_CPDAG(data_array_fmri_CPDAG, DAG_notears)
        res_notears.append(time_notears)
        res_list[4].append(res_notears)
        
        
        print("=====================")
        
        time_11 = time.perf_counter()
        # CPDAG_proposed_with_true = ori.identify_direction_begin_with_CPDAG_new_3(data_array_fmri_data_temp, data_array_fmri_CPDAG_original, 0.05, 0.01, 0.001)
        # CPDAG_proposed_with_true = ori.identify_direction_begin_with_CPDAG_new_3(data_array_fmri_data_temp, data_array_fmri_CPDAG, 0.05, 0.01, 0.001)
        CPDAG_proposed_with_true = ori.identify_direction_begin_with_CPDAG_new(data_array_fmri_data_temp, data_array_fmri_CPDAG_original, 0.05, 0.01, 0.01)
        CPDAG_proposed_with_true = utils_v2.detect_exceptions_2(CPDAG_proposed_with_true, collider_fmri_old)
        CPDAG_proposed_with_true = utils_v2.do_Meek_rule(CPDAG_proposed_with_true)

        time_12 = time.perf_counter()

        time_proposed_true = (time_12 - time_11)
        print(f"data: {data_array_fmri_data_temp[0]}")
        print(f"CPDAG_proposed_with_true: \n{CPDAG_proposed_with_true}")
        print(f"data_array_fmri_CPDAG: \n{data_array_fmri_CPDAG}")
        print(f"data_array_fmri_CPDAG_original: \n{data_array_fmri_CPDAG_original}")
        print(evaluation_CPDAG(data_array_fmri_CPDAG, CPDAG_proposed_with_true))
        print(time_proposed_true + time_PC)
        res_proposed_true = evaluation_CPDAG(data_array_fmri_CPDAG, CPDAG_proposed_with_true)
        res_proposed_true.append(time_proposed_true + time_PC)
        res_list[5].append(res_proposed_true)
        # draw.draw_CPDAG(CPDAG_proposed)
        print("======================")

        print("=====================")
        ##### sachs
        
        print(data_array_sachs)

        time_5 = time.perf_counter()
        cg = pc(data_array_sachs_data_temp, 0.05, uc_priority=0, indep_test="fisherz", stable=True)
        CPDAG_PC = cg.G.graph
        # print(CPDAG_PC[0][1])
        time_6 = time.perf_counter()
        time_PC = (time_6 - time_5)
        CPDAG_PC = change_matrix(CPDAG_PC)
        
        DAG_temp = np.where(CPDAG_PC == 1.0, 1.0, 0.0)
        arcs = utils.matrix_to_edge(DAG_temp)
        g = cd.DAG(arcs=arcs)

        collider_old = g.vstructures()

        print(f"data: {data_array_sachs_data_temp[0]}")
        print(CPDAG_PC)
        print(data_array_sachs_CPDAG)
        print(evaluation_CPDAG(data_array_sachs_CPDAG, CPDAG_PC))
        print(time_PC)
        res_PC = evaluation_CPDAG(data_array_sachs_CPDAG, CPDAG_PC)
        res_PC.append(time_PC)
        res_list_sachs[0].append(res_PC)
        # draw.draw_CPDAG(CPDAG_PC)
    
        print("=====================")
        time_3 = time.perf_counter()
        # CPDAG_PL = PL.PC_LiNGAM(data_array_sachs_data_temp, CPDAG_PC, 0.05)
        CPDAG_PL = PL.PC_LiNGAM_CPDAG(data_array_sachs_data_temp, CPDAG_PC, 0.05)
        CPDAG_PL = utils.do_Meek_rule(CPDAG_PL)
        time_4 = time.perf_counter()
        time_PL = (time_4 - time_3)
        print(f"data: {data_array_sachs_data_temp[0]}")
        print(f"CPDAG_PC: \n {CPDAG_PC}")
        print(CPDAG_PL)
        print(data_array_sachs_CPDAG)
        print(evaluation_CPDAG(data_array_sachs_CPDAG, CPDAG_PL))
        print(time_PL + time_PC)
        res_PL = evaluation_CPDAG(data_array_sachs_CPDAG, CPDAG_PL)
        res_PL.append(time_PL + time_PC)
        res_list_sachs[2].append(res_PL)
        # draw.draw_CPDAG(CPDAG_PL)

        
        # time_5 = time.perf_counter()
        # cg = pc(data_array_fmri_data, indep_test="kci")
        # CPDAG_PC = cg.G.graph
        # time_6 = time.perf_counter()
        # time_PC = (time_6 - time_5)
        # print(CPDAG_PC)
        # print(time_PC)

        print("=====================")
        # draw.draw_CPDAG(CPDAG_PC)
        time_1 = time.perf_counter()
        # CPDAG_proposed = ori.identify_direction_begin_with_CPDAG_new_3(data_array_sachs_data_temp, CPDAG_PC, 0.05, 0.01, 0.001)
        CPDAG_proposed = ori.identify_direction_begin_with_CPDAG_new(data_array_sachs_data_temp, CPDAG_PC, 0.05, 0.01, 0.001)
        # CPDAG_proposed = utils.do_Meek_rule(CPDAG_proposed)
        # print(f"data: {data_array_sachs_data_temp[0]}")
        # print(CPDAG_proposed)
        CPDAG_proposed = utils_v2.detect_exceptions_2(CPDAG_proposed, collider_old)
        CPDAG_proposed = utils_v2.do_Meek_rule(CPDAG_proposed)
        # draw.draw_CPDAG(CPDAG_proposed)
        # print("new: \n", CPDAG_proposed)
        # temp_flag = utils.detect_expections(CPDAG_proposed_new, collider_old)
        # if temp_flag:
        #     count_e_sachs.append(temp_flag)
        #     draw.draw_CPDAG(CPDAG_PC)
        #     draw.draw_CPDAG(CPDAG_proposed)
        #     draw.draw_CPDAG(CPDAG_proposed_new)
        time_2 = time.perf_counter()
        
        time_proposed = (time_2 - time_1)
        print(f"data: {data_array_sachs_data_temp[0]}")
        print(f"CPDAG_PC: \n {CPDAG_PC}")
        print(CPDAG_proposed)
        print(data_array_sachs_CPDAG)
        print(evaluation_CPDAG(data_array_sachs_CPDAG, CPDAG_proposed))
        print(time_proposed + time_PC)
        res_proposed = evaluation_CPDAG(data_array_sachs_CPDAG, CPDAG_proposed)
        res_proposed.append(time_proposed + time_PC)
        res_list_sachs[1].append(res_proposed)
        # draw.draw_CPDAG(CPDAG_proposed)
        
        print("=====================")
        time_7 = time.perf_counter()
        lingam_model = lingam.DirectLiNGAM()
        lingam_model.fit(data_array_sachs_data_temp)
        DAG_L = lingam_model.adjacency_matrix_
        DAG_L = np.where(DAG_L != 0.0, 1.0, 0.0)
        time_8 = time.perf_counter()
        time_L = (time_8 - time_7)
        for i in range(len(DAG_L)):
            for j in range(len(DAG_L)):
                if DAG_L[i][j] == 1.0:
                    DAG_L[j][i] = -1.0
        
        # print(f"data: {data_array_sachs_data_temp[0]}")
        # print(DAG_L)
        # print(data_array_sachs_CPDAG)
        # print(evaluation_CPDAG(data_array_sachs_CPDAG, DAG_L))
        # print(time_L)
        res_L = evaluation_CPDAG(data_array_sachs_CPDAG, DAG_L)
        res_L.append(time_L)
        res_list_sachs[3].append(res_L)
        # draw.draw_CPDAG(DAG_L)
        print("=====================")
        time_9 = time.perf_counter()
        W_est = notears_linear(data_array_sachs_data_temp, lambda1=0.01, loss_type='l2')
        DAG_notears = np.where(W_est != 0.0, 1.0, 0.0)
        time_10 = time.perf_counter()
        time_notears = (time_10 - time_9)
        # print(DAG_L)
        for i in range(len(DAG_notears)):
            for j in range(len(DAG_notears)):
                if DAG_notears[i][j] == 1.0:
                    DAG_notears[j][i] = -1.0
        # print(f"data: {data_array_sachs_data_temp[0]}")
        # print(DAG_notears)
        # print(data_array_fmri_CPDAG)
        # print(evaluation_CPDAG(data_array_sachs_CPDAG, DAG_notears))
        # print(time_notears)

        res_notears = evaluation_CPDAG(data_array_sachs_CPDAG, DAG_notears)
        res_notears.append(time_notears)
        res_list_sachs[4].append(res_notears)
        
                
        print("======================")
        
        time_11 = time.perf_counter()
        # CPDAG_proposed_with_true = ori.identify_direction_begin_with_CPDAG_new(data_array_sachs_data_temp, data_array_sachs_CPDAG_original, 0.05, 0.01, 0.001)
        CPDAG_proposed_with_true = ori.identify_direction_begin_with_CPDAG_new(data_array_sachs_data_temp, data_array_sachs_CPDAG_original, 0.05, 0.01, 0.001)
        CPDAG_proposed_with_true = utils_v2.detect_exceptions_2(CPDAG_proposed_with_true, collider_sachs_old)
        CPDAG_proposed_with_true = utils_v2.do_Meek_rule(CPDAG_proposed_with_true)

        time_12 = time.perf_counter()

        time_proposed_true = (time_12 - time_11)
        print(f"data: {data_array_sachs_data_temp[0]}")
        print(f"CPDAG_proposed_with_true: \n{CPDAG_proposed_with_true}")
        print(f"data_array_sachs_CPDAG: \n{data_array_sachs_CPDAG}")
        print(f"data_array_sachs_CPDAG_original: \n{data_array_sachs_CPDAG_original}")
        print(evaluation_CPDAG(data_array_sachs_CPDAG, CPDAG_proposed_with_true))
        print(time_proposed_true + time_PC)
        res_proposed_true = evaluation_CPDAG(data_array_sachs_CPDAG, CPDAG_proposed_with_true)
        res_proposed_true.append(time_proposed_true + time_PC)
        res_list_sachs[5].append(res_proposed_true)
        # draw.draw_CPDAG(CPDAG_proposed)
        print("=====================")
        
        

    write_excel(res_list, "fmri_res_f"+formatted_time)
    write_excel(res_list_sachs, "sachs_res_f"+formatted_time)

    # print("fMRI: ", count_e_fmri)
    # print("Sachs: ", count_e_sachs)