import random
import pandas as pd
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
from causallearn.search.ConstraintBased.PC import pc

import time

import bnlearn as bn
import xlwt

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


def write_excel(res_list, name):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet('PC',cell_overwrite_ok=True)
    sheet2 = f.add_sheet('proposed',cell_overwrite_ok=True)
    sheet3 = f.add_sheet('PL',cell_overwrite_ok=True)
    sheet4 = f.add_sheet('LiNGAM',cell_overwrite_ok=True)
    
    res_PC = res_list[0]
    res_proposed = res_list[1]
    res_PL = res_list[2]
    res_L = res_list[3]
    
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

if __name__ == '__main__':
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
    
    residuals_list = PL.get_residual_for_one_DAG(data_array_fmri_data, data_array_fmri)
    gaussian_list = list()
    change_list_fmri = list()
    index = 0
    for r in residuals_list:
        if utils.is_Gaussian(r, 0.01):
            gaussian_list.append(1)
        else:
            gaussian_list.append(0)
            change_list_fmri.append(index)
        index += 1
    print(gaussian_list)
    print(change_list_fmri)


    res_CPDAG = utils.get_True_CPDAG(data_array_fmri, change_list_fmri)
    print(data_array_fmri)
    print(res_CPDAG)

    draw.draw_directed_graph(data_array_fmri, change_list_fmri)
    draw.draw_CPDAG(res_CPDAG, change_list_fmri)

    residuals_list = PL.get_residual_for_one_DAG(data_array_sachs_data, data_array_sachs)
    gaussian_list = list()
    change_list_sachs = list()
    index = 0
    for r in residuals_list:
        if utils.is_Gaussian(r, 0.05):
            gaussian_list.append(1)
        else:
            gaussian_list.append(0)
            change_list_sachs.append(index)
        index += 1
    print(gaussian_list)
    print(change_list_sachs)
    res_CPDAG = utils.get_True_CPDAG(data_array_sachs, change_list_sachs)
    print(data_array_sachs)
    print(res_CPDAG)

    draw.draw_directed_graph(data_array_sachs, change_list_sachs)
    draw.draw_CPDAG(res_CPDAG, change_list_sachs)


    # current_time = time.localtime()
    # formatted_time = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    # # alpha_l_list = [0.01, 0.05, 0.1, 0.5]
    # # alpha_i_list = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    # # alpha_l_list = [0.01, 0.05, 0.1, 0.5]
    # # alpha_i_list = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    # # alpha_p = [1]

    # # samples = [25, 50, 100]

    # df_fmri = pd.read_excel("fMRI.xlsx", sheet_name="Sheet1", header=None)

    # data_array_fmri = df_fmri.to_numpy()
    # # print(data_array)
    # data_array_fmri = np.where(df_fmri.to_numpy() == -1, 0, df_fmri.to_numpy())
    # data_array_fmri = np.where(data_array_fmri != 0, 1.0, 0.0)
    # data_array_fmri = data_array_fmri.T
    # # print(data_array_fmri.T)

    # # draw.draw(data_array_fmri.T)

    # df_fmri_data = pd.read_excel("fMRI.xlsx", sheet_name="Sheet2", header=None)
    # data_array_fmri_data = df_fmri_data.to_numpy()
    # # indices = np.random.choice(data_array_fmri_data.shape[0], 100, replace=False)
    # # selected_data_array_fmri_data = data_array_fmri_data[indices, : ]
    
    # print(data_array_fmri_data.shape)
    # # print(selected_data_array_fmri_data.shape)

    
    # df_sachs_data = pd.read_excel("sachs.xls", sheet_name="Sheet2")
    # data_array_sachs_data = df_sachs_data.to_numpy()
    # # indices = np.random.choice(data_array_sachs_data.shape[0], 100, replace=False)
    # # selected_data_array_sachs_data = data_array_sachs_data[indices, : ]
    # print(data_array_sachs_data)
    # # print(selected_data_array_sachs_data.shape)
    
    # # model = bn.import_DAG('sachs')
    # # row_order = ["PKC","PKA", "Raf", "Jnk", "P38", "Mek", "Erk", "Akt", "Plcg", "PIP3", "PIP2"]
    # # col_order = ["PKC","PKA", "Raf", "Jnk", "P38", "Mek", "Erk", "Akt", "Plcg", "PIP3", "PIP2"]
    # # # print(model["adjmat"])
    # # df_sachs = model["adjmat"]
    # # df_sachs = df_sachs.reindex(index=row_order, columns=col_order)
    # # print(df_sachs)
    # # data_array_sachs = np.where(df_sachs.to_numpy(), 1.0, 0.0)
    # # print(data_array_sachs.T)
    # data_array_sachs=np.array(
    #     [
    #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    #         ]
    #     )
    # # data_array_sachs
    
    # # draw.draw(data_array_sachs.T)
    # # print(model["adjmat"][2:,1:], type(model["adjmat"]))

    # # ex.execute_KCI_once([df_fmri, ])
    # # data_array_sachs_data_copy = data_array_sachs_data
    # # data_array_fmri_data_copy = data_array_fmri_data
    
    # count = 0
    # res_list = [[] for i in range(4)]
    # res_list_sachs = [[] for i in range(4)]
    # for _ in range(50):
    #     indices = np.random.choice(data_array_sachs_data.shape[0], 100, replace=False)
    #     data_array_sachs_data_temp = data_array_sachs_data[indices, : ]
    #     indices = np.random.choice(data_array_fmri_data.shape[0], 100, replace=False)
    #     data_array_fmri_data_temp = data_array_fmri_data[indices, : ]
        
    #     count += 1
    #     current_time = time.localtime()
    #     formatted_time = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        
    #     ##### fMRI
    #     print(data_array_fmri)
    #     time_5 = time.perf_counter()
    #     cg = pc(data_array_fmri_data_temp, uc_priority=0, indep_test="fisherz", stable=True)
    #     CPDAG_PC = cg.G.graph
    #     # print(CPDAG_PC[0][1])
    #     time_6 = time.perf_counter()
    #     time_PC = (time_6 - time_5)
    #     CPDAG_PC = change_matrix(CPDAG_PC)
    #     print(CPDAG_PC)
    #     print(evaluation(data_array_fmri, CPDAG_PC))
    #     print(time_PC)
    #     res_PC = evaluation(data_array_fmri, CPDAG_PC)
    #     res_PC.append(time_PC)
    #     res_list[0].append(res_PC)
    #     # draw.draw_CPDAG(CPDAG_PC)
        
    #     time_1 = time.perf_counter()
    #     CPDAG_proposed = ori.identify_direction_begin_with_CPDAG_new(data_array_fmri_data_temp, CPDAG_PC, 0.05, 0.01, 0.001)
    #     # CPDAG_proposed = utils.do_Meek_rule(CPDAG_proposed)
    #     time_2 = time.perf_counter()
        
    #     time_proposed = (time_2 - time_1)
    #     print(CPDAG_proposed)
    #     print(evaluation(data_array_fmri, CPDAG_proposed))
    #     print(time_proposed + time_PC)
    #     res_proposed = evaluation(data_array_fmri, CPDAG_proposed)
    #     res_proposed.append(time_proposed + time_PC)
    #     res_list[1].append(res_proposed)

    #     time_3 = time.perf_counter()
    #     CPDAG_PL = PL.PC_LiNGAM(data_array_fmri_data_temp, CPDAG_PC, 0.05)
    #     CPDAG_PL = utils.do_Meek_rule(CPDAG_PL)
    #     time_4 = time.perf_counter()
    #     time_PL = (time_4 - time_3)
    #     print(CPDAG_PL)
    #     print(evaluation(data_array_fmri, CPDAG_PL))
    #     print(time_PL + time_PC)
    #     res_PL = evaluation(data_array_fmri, CPDAG_PL)
    #     res_PL.append(time_PL + time_PC)
    #     res_list[2].append(res_PL)
        
    #     # time_5 = time.perf_counter()
    #     # cg = pc(data_array_fmri_data, indep_test="kci")
    #     # CPDAG_PC = cg.G.graph
    #     # time_6 = time.perf_counter()
    #     # time_PC = (time_6 - time_5)
    #     # print(CPDAG_PC)
    #     # print(time_PC)

    #     time_7 = time.perf_counter()
    #     lingam_model = lingam.DirectLiNGAM()
    #     lingam_model.fit(data_array_fmri_data_temp)
    #     DAG_L = lingam_model.adjacency_matrix_
    #     DAG_L = np.where(DAG_L != 0.0, 1.0, 0.0)
    #     time_8 = time.perf_counter()
    #     time_L = (time_8 - time_7)
    #     print(DAG_L)
    #     print(evaluation(data_array_fmri, DAG_L))
    #     print(time_L)

    #     res_L = evaluation(data_array_fmri, DAG_L)
    #     res_L.append(time_L + time_PC)
    #     res_list[3].append(res_L)

    #     ##### sachs
    #     print(data_array_sachs)
    #     time_5 = time.perf_counter()
    #     cg = pc(data_array_sachs_data_temp, uc_priority=0, indep_test="fisherz",stable=True)
    #     CPDAG_PC = cg.G.graph
    #     # print(CPDAG_PC[0][1])
    #     time_6 = time.perf_counter()
    #     time_PC = (time_6 - time_5)
    #     CPDAG_PC = change_matrix(CPDAG_PC)
    #     print(CPDAG_PC)
    #     print(evaluation(data_array_sachs, CPDAG_PC))
    #     print(time_PC)
    #     res_PC = evaluation(data_array_sachs, CPDAG_PC)
    #     res_PC.append(time_PC)
    #     res_list_sachs[0].append(res_PC)
    #     # draw.draw_CPDAG(CPDAG_PC)
        
    #     time_1 = time.perf_counter()
    #     CPDAG_proposed = ori.identify_direction_begin_with_CPDAG_new(data_array_sachs_data_temp, CPDAG_PC, 0.05, 0.01, 0.001)
    #     CPDAG_proposed = utils.do_Meek_rule(CPDAG_proposed)
    #     time_2 = time.perf_counter()
        
    #     time_proposed = (time_2 - time_1)
    #     print(CPDAG_proposed)
    #     print(evaluation(data_array_sachs, CPDAG_proposed))
    #     print(time_proposed + time_PC)
    #     res_proposed = evaluation(data_array_sachs, CPDAG_proposed)
    #     res_proposed.append(time_proposed + time_PC)
    #     res_list_sachs[1].append(res_proposed)

    #     time_3 = time.perf_counter()
    #     CPDAG_PL = PL.PC_LiNGAM(data_array_sachs_data_temp, CPDAG_PC, 0.05)
    #     CPDAG_PL = utils.do_Meek_rule(CPDAG_PL)
    #     time_4 = time.perf_counter()
    #     time_PL = (time_4 - time_3)
    #     print(CPDAG_PL)
    #     print(evaluation(data_array_sachs, CPDAG_PL))
    #     print(time_PL + time_PC)
    #     res_PL = evaluation(data_array_sachs, CPDAG_PL)
    #     res_PL.append(time_PL + time_PC)
    #     res_list_sachs[2].append(res_PL)
        
    #     # time_5 = time.perf_counter()
    #     # cg = pc(data_array_fmri_data, indep_test="kci")
    #     # CPDAG_PC = cg.G.graph
    #     # time_6 = time.perf_counter()
    #     # time_PC = (time_6 - time_5)
    #     # print(CPDAG_PC)
    #     # print(time_PC)

    #     time_7 = time.perf_counter()
    #     lingam_model = lingam.DirectLiNGAM()
    #     lingam_model.fit(data_array_sachs_data_temp)
    #     DAG_L = lingam_model.adjacency_matrix_
    #     DAG_L = np.where(DAG_L != 0.0, 1.0, 0.0)
    #     time_8 = time.perf_counter()
    #     time_L = (time_8 - time_7)
    #     print(DAG_L)
    #     print(evaluation(data_array_sachs, DAG_L))
    #     print(time_L)
    #     res_L = evaluation(data_array_sachs, DAG_L)
    #     res_L.append(time_L + time_PC)
    #     res_list_sachs[3].append(res_L)
    # write_excel(res_list, "fmri_res_f")
    # write_excel(res_list_sachs, "sachs_res_f")