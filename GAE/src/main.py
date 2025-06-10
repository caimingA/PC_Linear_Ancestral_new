#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python codes for 'A Graph Autoencoder Approach to Causal Structure Learning', NeurIPS 2019 Workshop
Authors: Ignavier Ng*, University of Toronto
         Shengyu Zhu, Huawei Noah's Ark Lab,
         Zhitang Chen, Huawei Noah's Ark Lab,
         Zhuangyan Fang*, Peking University
         * Work was done during an internship at Huawei Noah's Ark Lab
"""

import logging
from pytz import timezone
from datetime import datetime
import numpy as np

from data_loader import SyntheticDataset
from models import GAE
from trainers import ALTrainer
from helpers.config_utils import save_yaml_config, get_args
from helpers.log_helper import LogHelper
from helpers.tf_utils import set_seed
from helpers.dir_utils import create_dir
from helpers.analyze_utils import count_accuracy, plot_recovered_graph

# import bnlearn as bn
import xlwt
import sys
sys.path.append(r"C:\code_old\code\python\PC_Linear_Ancestral_experiment_4")
# import utils
import time
import pandas as pd

def write_excel(res_list, name):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet('GAE',cell_overwrite_ok=True)
    # sheet1 = f.add_sheet('PC',cell_overwrite_ok=True)
    # sheet2 = f.add_sheet('proposed',cell_overwrite_ok=True)
    # sheet3 = f.add_sheet('PL',cell_overwrite_ok=True)
    # sheet4 = f.add_sheet('LiNGAM',cell_overwrite_ok=True)
    # sheet5 = f.add_sheet('NoTears',cell_overwrite_ok=True)
    # sheet6 = f.add_sheet('RCD',cell_overwrite_ok=True)

    res_PC = res_list[0]
    # res_proposed = res_list[1]
    # res_PL = res_list[2]
    # res_L = res_list[3]
    # res_notears = res_list[4]
    # res_rcd = res_list[5]
    
    for i in range(len(res_PC)):
        for j in range(len(res_PC[0])):
            sheet1.write(i,j, res_PC[i][j])

    # for i in range(len(res_proposed)):
    #     for j in range(len(res_proposed[0])):
    #         sheet2.write(i,j, res_proposed[i][j])

    # for i in range(len(res_PL)):
    #     for j in range(len(res_PL[0])):
    #         sheet3.write(i,j, res_PL[i][j])

    # for i in range(len(res_L)):
    #     for j in range(len(res_L[0])):
    #         sheet4.write(i,j, res_L[i][j])

    # for i in range(len(res_notears)):
    #     for j in range(len(res_notears[0])):
    #         sheet5.write(i,j, res_notears[i][j])

    # for i in range(len(res_notears)):
    #     for j in range(len(res_notears[0])):
    #         sheet6.write(i,j, res_rcd[i][j])

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
    pre = correct/estimated_edges
    rec = correct/true_edges
    F = 0
    if pre + rec == 0:
        F = 0
    else: 
        F = 2*pre*rec / (pre + rec)
    return [pre, rec, F]


# def main():
#     # Get arguments parsed
#     args = get_args()

#     # Setup for logging
#     output_dir = 'output/{}'.format(datetime.now(timezone('Asia/Hong_Kong')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
#     create_dir(output_dir)
#     LogHelper.setup(log_path='{}/training.log'.format(output_dir),
#                     level_str='INFO')
#     _logger = logging.getLogger(__name__)

#     # Save the configuration for logging purpose
#     save_yaml_config(args, path='{}/config.yaml'.format(output_dir))

#     # Reproducibility
#     set_seed(args.seed)

#     # Get dataset
#     dataset = SyntheticDataset(args.n, args.d, args.graph_type, args.degree, args.sem_type,
#                                args.noise_scale, args.dataset_type, args.x_dim)
#     _logger.info('Finished generating dataset')
#     # _logger.info(dataset)
#     _logger.info(dataset.X.shape)
#     _logger.info(dataset.W)
#     model = GAE(args.n, args.d, args.x_dim, args.seed, args.num_encoder_layers, args.num_decoder_layers,
#                 args.hidden_size, args.latent_dim, args.l1_graph_penalty, args.use_float64)
#     model.print_summary(print_func=model.logger.info)

#     trainer = ALTrainer(args.init_rho, args.rho_thres, args.h_thres, args.rho_multiply,
#                         args.init_iter, args.learning_rate, args.h_tol,
#                         args.early_stopping, args.early_stopping_thres)
#     W_est = trainer.train(model, dataset.X, dataset.W, args.graph_thres,
#                           args.max_iter, args.iter_step, output_dir)
#     _logger.info('Finished training model')

#     # Save raw recovered graph, ground truth and observational data after training
#     np.save('{}/true_graph.npy'.format(output_dir), dataset.W)
#     np.save('{}/observational_data.npy'.format(output_dir), dataset.X)
#     np.save('{}/final_raw_recovered_graph.npy'.format(output_dir), W_est)

#     # Plot raw recovered graph
#     plot_recovered_graph(W_est, dataset.W,
#                          save_name='{}/raw_recovered_graph.png'.format(output_dir))

#     _logger.info('Filter by constant threshold')
#     W_est = W_est / np.max(np.abs(W_est))    # Normalize

#     # Plot thresholded recovered graph
#     W_est[np.abs(W_est) < args.graph_thres] = 0    # Thresholding
#     plot_recovered_graph(W_est, dataset.W,
#                          save_name='{}/thresholded_recovered_graph.png'.format(output_dir))
#     results_thresholded = count_accuracy(dataset.W, W_est)
#     _logger.info('Results after thresholding by {}: {}'.format(args.graph_thres, results_thresholded))
def main(p, n, data, W):
    # Get arguments parsed
    args = get_args()

    # Setup for logging
    output_dir = 'output/{}'.format(datetime.now(timezone('Asia/Hong_Kong')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
    create_dir(output_dir)
    LogHelper.setup(log_path='{}/training.log'.format(output_dir),
                    level_str='INFO')
    _logger = logging.getLogger(__name__)

    # Save the configuration for logging purpose
    save_yaml_config(args, path='{}/config.yaml'.format(output_dir))

    # Reproducibility
    set_seed(args.seed)

    # Get dataset
    # dataset = SyntheticDataset(args.n, args.d, args.graph_type, args.degree, args.sem_type,
                            #    args.noise_scale, args.dataset_type, args.x_dim)
    # print(dataset)
    # _logger.info('Finished generating dataset')

    model = GAE(n, p, 1, args.seed, args.num_encoder_layers, args.num_decoder_layers,
                args.hidden_size, 1, args.l1_graph_penalty, args.use_float64)
    model.print_summary(print_func=model.logger.info)

    trainer = ALTrainer(args.init_rho, args.rho_thres, args.h_thres, args.rho_multiply,
                        args.init_iter, args.learning_rate, args.h_tol,
                        args.early_stopping, args.early_stopping_thres)
    W_est = trainer.train(model, data, W, args.graph_thres,
                          args.max_iter, args.iter_step, output_dir)
    W_est = W_est / np.max(np.abs(W_est))    # Normalize

    W_est[np.abs(W_est) < args.graph_thres] = 0    # Thresholding
    
    return W_est


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
    
    # print(data_array_fmri_data.shape)
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
    # data_array_fmri_CPDAG = utils.get_True_CPDAG(data_array_fmri, [2])
    
    data_array_fmri_CPDAG = np.array([
        [ 0., -1.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
        [-1.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  1.,  0., -1.,  0.,  0.,  0., -1.,  0.,  0.],
        [ 0.,  0.,  1.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0., -1.],
        [ 0.,  0.,  0.,  0.,  0., -1.,  0., -1.,  0.,  0.],
        [ 0.,  0.,  1.,  0.,  0.,  0.,  1.,  0., -1.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0., -1.],
        [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.]
])
    # print(data_array_fmri_CPDAG)
    # input()
    
    df_sachs_data = pd.read_excel("sachs.xls", sheet_name="Sheet2")
    data_array_sachs_data = df_sachs_data.to_numpy()
    # indices = np.random.choice(data_array_sachs_data.shape[0], 100, replace=False)
    # selected_data_array_sachs_data = data_array_sachs_data[indices, : ]
    # print(data_array_sachs_data)
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

    # data_array_sachs_CPDAG = utils.get_True_CPDAG(data_array_sachs, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    data_array_sachs_CPDAG = data_array_sachs_CPDAG = np.array([
    [ 0., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.],
    [ 1.,  0., -1., -1., -1., -1., -1., -1.,  0.,  0.,  0.],
    [ 1.,  1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
    [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [ 1.,  1.,  1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.],
    [ 0.,  1.,  0.,  0.,  0.,  1.,  0., -1.,  0.,  0.,  0.],
    [ 0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1.],
    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0., -1.],
    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.]
    ])
    
    data_array_fmri_CPDAG.flags.writeable = False
    data_array_sachs_CPDAG.flags.writeable = False
    # print(data_array_sachs_CPDAG)
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
    res_list = [[] for i in range(6)]
    res_list_sachs = [[] for i in range(6)]
    count_e_fmri = list()
    count_e_sachs = list()
    # W_est= main()
    # np.random.seed(2025)
    for _ in range(50):
        indices = np.random.choice(data_array_sachs_data.shape[0], 200, replace=False)
        data_array_sachs_data_temp = data_array_sachs_data[indices, : ].reshape(200, 11, 1)
        indices = np.random.choice(data_array_fmri_data.shape[0], 100, replace=False)
        data_array_fmri_data_temp = data_array_fmri_data[indices, : ].reshape(100, 10, 1)
        
        count += 1
        current_time = time.localtime()
        formatted_time = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        ##### fMRI
        print("=====================fMRI", count, "fMRI=====================")
        time_1 = time.perf_counter()
        W_est= main(n=100, p=len(data_array_fmri), data=data_array_fmri_data_temp, W=data_array_fmri)
        time_2 = time.perf_counter()
        time_GAE = (time_2 - time_1)
        W_est = np.where(W_est != 0.0, 1.0, 0.0)
        for i in range(len(W_est)):
            for j in range(len(W_est)):
                if W_est[i][j] == 1.0:
                    W_est[j][i] = -1.0
        res_GAE = evaluation_CPDAG(data_array_fmri_CPDAG, W_est)
        res_GAE.append(time_GAE)
        res_list[0].append(res_GAE)

        ##### sachs
        print("=====================sachs", count, "sachs=====================")
        time_1 = time.perf_counter()
        W_est= main(n=200, p=len(data_array_sachs), data=data_array_sachs_data_temp, W=data_array_sachs)
        time_2 = time.perf_counter()
        time_GAE = (time_2 - time_1)
        W_est = np.where(W_est != 0.0, 1.0, 0.0)
        for i in range(len(W_est)):
            for j in range(len(W_est)):
                if W_est[i][j] == 1.0:
                    W_est[j][i] = -1.0
        res_GAE = evaluation_CPDAG(data_array_sachs_CPDAG, W_est)
        res_GAE.append(time_GAE)
        res_list_sachs[0].append(res_GAE)
    write_excel(res_list, "fmri_res_f"+formatted_time)
    write_excel(res_list_sachs, "sachs_res_f"+formatted_time)