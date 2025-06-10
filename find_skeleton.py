# import numpy as np
# import causaldag as cd
# import utils
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt


# def extract_colliders(colliders):
#     collider_dict = dict()
#     collider_midnode_list = list()
#     for col in colliders:
#         # for i in range(3):
#         i = col[0]
#         j = col[2]
#         mid_node = col[1]

#         collider_midnode_list.append(mid_node)

#         if mid_node in collider_dict:
#             if i not in collider_dict[mid_node]:
#                 collider_dict[mid_node].append(i)
#             if j not in collider_dict[mid_node]:
#                 collider_dict[mid_node].append(j)
#         else:
#             collider_dict[mid_node] = [i, j]
        
#     return collider_dict, collider_midnode_list


# # def make_except_nodes(collider_midnode_list):



# def extract_skeleton_DFS(CPDAG, collider_dict, collider_midnode_list):
#     node_num = len(CPDAG)
#     DFS_stack = list()
    
#     skeleton_list_list = list()
    

#     collider_midnode_num = len(collider_midnode_list)
#     visit_list = np.arange(collider_midnode_num)
#     # print(visit_list)
#     np.random.shuffle(visit_list)
#     # print(visit_list)

#     start_node = visit_list[0]
#     DFS_stack.append(start_node)

#     skeleton_list = list()
#     while len(DFS_stack) != 0:
#         ptr = DFS_stack.pop()

#         for i in range(node_num):

#             if CPDAG[ptr][i] == -1 and CPDAG[i][ptr] == -1:
#                 DFS_stack.append(i)
#                 skeleton_list.append(i)
#             if CPDAG[ptr][i] == 1 and CPDAG[i][ptr]:
#                 DFS_stack.append(i)
#                 skeleton_list.append(i)
    

# def extract_Gaussian_and_nonGaussian(data, s_alpha):
#     node_num = len(data[0])
#     gaussian_list = list()
#     nonGaussian_list = list()

#     print("******************")
#     for i in range(node_num):
#         print(i)
#         # plt.hist(data[: , i], bins=50)
#         # plt.show()
#         if utils.is_Gaussian(data[:, i], s_alpha):
#             gaussian_list.append(i)
#         else:
#             nonGaussian_list.append(i)

#     return np.array(gaussian_list), np.array(nonGaussian_list)


# def regress_nonGaussian_on_Gaussian(data, gaussian_list, nonGaussian_list):
#     if len(gaussian_list) == 0:
#         return data
    
#     data_Gaussian = list()
#     data_nonGaussian = list()
#     node_num = len(data[0])
#     node_num_ga = len(gaussian_list)
#     node_num_ng = len(nonGaussian_list)
#     i = 0
#     j = 0
#     count = 0
#     while i + j != node_num:
#         if i >= node_num_ga:
#             data_nonGaussian.append(data[:, count])
#             j += 1
#             count += 1
#             continue
#         if j >= node_num_ng:
#             data_Gaussian.append(data[:, count])
#             i += 1
#             count += 1
#             continue

#         if gaussian_list[i] < nonGaussian_list[j]:
#             data_Gaussian.append(data[:, count])
#             i += 1
#             count += 1
#             continue
#         if gaussian_list[i] > nonGaussian_list[j]:
#             data_nonGaussian.append(data[:, count])
#             j += 1
#             count += 1
#             continue
        
        

    
#     # for i in gaussian_list:
#     #     data_Gaussian.append(data[:, i])

#     data_Gaussian = np.array(data_Gaussian).T

#     # for i in nonGaussian_list:
#     #     data_nonGaussian.append(data[:, i])

#     data_nonGaussian = np.array(data_nonGaussian).T

#     residual_nonGaussian = list()

#     reg = LinearRegression(fit_intercept=False)
    
#     for i in range(node_num_ng):
#         res = reg.fit(data_Gaussian, data_nonGaussian[:, i]) # 自变量，因变量
#         residual_nonGaussian.append(data_nonGaussian[:, i] - reg.predict(data_Gaussian))
#         # coef = res.coef_
#         # residual_nonGaussian.append(data_nonGaussian[:, i] - np.dot(coef, data_Gaussian.T).T)
#     residual_nonGaussian = np.array(residual_nonGaussian).T
#     # res = reg.fit(data_Gaussian, data_nonGaussian) # 自变量，因变量
#     # coef = res.coef_

#     # residual_nonGaussian = data_nonGaussian - np.dot(coef, data_Gaussian)
#     # residual_nonGaussian = data_nonGaussian - np.dot(coef, data_Gaussian.T).T

#     return residual_nonGaussian


# # x_0与nonGaussian_list里都是真实节点的index
# def regress_nonGaussian_on_x0(data, x_0, nonGaussian_list):
#     node_num = len(nonGaussian_list)
#     data_nonGaussian = list()
#     x_0_data = list()
#     for i in range(node_num):
#         if nonGaussian_list[i] == x_0:
#             x_0_data.append(data[:, i])
#             x_0_data=data[:, i]
#         else:
#             data_nonGaussian.append(data[:, i])
#     # for i in nonGaussian_list:
#     #     data_nonGaussian.append(data[:, i])
#     x_0_data = np.array(x_0_data)
#     data_nonGaussian = np.array(data_nonGaussian).T

#     reg = LinearRegression(fit_intercept=False)
#     # res = reg.fit(x_0_data.reshape(-1, 1), data_nonGaussian) # 自变量，因变量
#     # coef = res.coef_

#     residual_nonGaussian = list()
#     for i in range(len(data_nonGaussian[0])):
#         res = reg.fit(x_0_data.reshape(-1, 1), data_nonGaussian[:, i]) # 自变量，因变量
#         residual_nonGaussian.append(data_nonGaussian[:, i] - reg.predict(x_0_data.reshape(-1, 1)))
#         # coef = res.coef_
#         # residual_nonGaussian.append(data_nonGaussian[:, i] - np.dot(coef, x_0_data))
#         # residual_nonGaussian.append(data_nonGaussian[:, i] - coef*x_0_data)
#     residual_nonGaussian = np.array(residual_nonGaussian).T
#     # residual_nonGaussian = data_nonGaussian - np.dot(coef, x_0.T).T

#     return residual_nonGaussian