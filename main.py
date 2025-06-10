import random
import numpy as np
import data_generator as dg
import find_ancestor as fa
import utils
import find_skeleton as fs
import copy
from multiprocessing import Pool

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


if __name__== '__main__':
    # random.seed(1)
    process_num = 10
    experiment_num = 50
    node_min = 5
    node_max = 10
    # sample_size = 1500
    sample_list = [1500, 2000, 3000, 5000, 10000]
    # sample_list = [10000]
    for n in range(6, 7):
        print("---------------# node = ", n, "---------------")   
        for s in sample_list:
            print("---------------# sample = ", s, "---------------")
            edges = (n - 1)*n/2
            node_list = [i for i in range(n)]
            eva_proposed = np.zeros((4, 4))
            eva_PL = np.zeros((4, 4))
            time_proposed = 0
            time_PL = 0
            error_count_proposed = 0
            error_count_PL = 0
            experiment_list = list()

            true_CPDAG_list = list()            

            for _i in range(experiment_num):    
                print("---------------", _i, "---------------")
                change_num = np.random.randint(n//3, n)
                change_list = random.sample(node_list, k=change_num)
                # edge_num = np.random.randint(0, edges)
                # print(edge_num)
                DAG_temp = dg.lower_triangle_graph(n, edges, n)
                # print(DAG_temp)
                # print(DAG_temp)
                # print(change_list)
                B_temp = dg.get_B_0(DAG_temp)

                noisy_temp = dg.get_noisy_Mix(s, DAG_temp, change_list)

                x_list_temp = dg.get_x(DAG_temp, noisy_temp, B_temp).T

                true_CPDAG = utils.get_True_CPDAG(DAG_temp, change_list)
                
                experiment_list.append([x_list_temp, DAG_temp, 0.05, 0.01, 0.001])

                true_CPDAG_list.append(true_CPDAG)
                
            with Pool(process_num) as p:
                res_list_proposed = p.map(ori.identify_direction_begin_with_CPDAG_multithread, experiment_list)
                # res_list_PL = p.map(PL.PC_LiNGAM_multithread, experiment_list)
            # print("proposed finished!")
            
            with Pool(process_num) as p:
                res_list_PL = p.map(PL.PC_LiNGAM_multithread, experiment_list)
            # print("PL finished!")
            # print(res_list_proposed)
            # res_list_proposed = np.array(res_list_proposed)
            # res_list_PL = np.array(res_list_PL)

            for _i in range(experiment_num):
                res_proposed = np.array(res_list_proposed[_i], dtype=object)
                res_PL = np.array(res_list_PL[_i], dtype=object)
                
                # print("CPDAG", res_proposed[0])
                # print("time", res_proposed[1])
                if np.mean(res_proposed[0] - true_CPDAG_list[_i]) != 0:
                    error_count_proposed += 1
                    print(true_CPDAG_list[_i])
                    print(res_proposed[0])
                eva_proposed += utils.evaluate(res_proposed[0], true_CPDAG_list[_i])
                if np.mean(res_PL[0] - true_CPDAG_list[_i]) != 0:
                    error_count_PL += 1
                    print(true_CPDAG_list[_i])
                    print(res_PL[0])
                eva_PL += utils.evaluate(res_PL[0], true_CPDAG_list[_i])

                time_proposed += res_proposed[1]
                print("time proposed: ", res_proposed[1])
                # print("proposed: ", res_proposed)
                time_PL += res_PL[1]
                print("time PL: ", res_PL[1])
                # print("PL: ", res_PL)
                # time_1 = time.time()
                # CPDAG_proposed = ori.identify_direction_begin_with_CPDAG_new(x_list_temp, DAG_temp, 0.05, 0.01, 0.001)
                # CPDAG_proposed = utils.do_Meek_rule(CPDAG_proposed)
                # time_2 = time.time()
                # # print(CPDAG_proposed)
                # eva_proposed += utils.evaluate(CPDAG_proposed, true_CPDAG)
                # print("proposed finished")
                # time_proposed += (time_2 - time_1)

                # time_3 = time.time()
                # CPDAG_PL = PL.PC_LiNGAM(x_list_temp, DAG_temp, 0.05)
                # CPDAG_PL = utils.do_Meek_rule(CPDAG_PL)
                # time_4 = time.time()
                # time_PL += (time_4 - time_3)
                # # print(CPDAG_PL)
                # eva_PL += utils.evaluate(CPDAG_PL, true_CPDAG)
                # print("PL finished")

                # if np.mean(CPDAG_proposed - true_CPDAG) != 0:
                #     print("true CPDAG")
                #     print(true_CPDAG)
                #     print("change list")
                #     print(change_list)
                #     print("CPDAG_proposed")
                #     print(CPDAG_proposed)
                #     print("CPDAG_PL")
                #     print(CPDAG_PL)
                #     error_count_proposed += 1                
                # if np.mean(CPDAG_PL - true_CPDAG) != 0:
                #     print("true CPDAG")
                #     print(true_CPDAG)
                #     print("change list")
                #     print(change_list)
                #     print("CPDAG_proposed")
                #     print(CPDAG_proposed)
                #     print("CPDAG_PL")
                #     print(CPDAG_PL)
                #     error_count_PL += 1
            we.write_res_excel("node_num_" + str(n) + "_sample_size_" + str(s), eva_proposed, eva_PL, time_proposed, time_PL, error_count_proposed, error_count_PL)