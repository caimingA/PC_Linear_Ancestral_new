import numpy as np
# from numba import jit


# @jit(nopython=True)
def get_noisy(T, matrix):
    num = len(matrix)
    # 均匀分布
    # noisy = np.random.random(size = (T, len(matrix)))*2
    # noisy = np.where(noisy > 1.0, noisy+1.0, noisy-3.0)

    # 指数分布
    # noisy = np.random.exponential(scale=1, size=(T, num))
    # noisy -= np.mean(noisy, axis=0)
    # 对数高斯分布
    noisy = np.random.lognormal(mean=0, sigma=1, size=(T, num))
    noisy -= np.e**(1/2)
    # noisy -= np.mean(noisy, axis=0)

    # 双高斯
    # mu1, sigma1 = 3, 1     # 第一个高斯分布的均值和标准差
    # mu2, sigma2 = -3, 1   # 第二个高斯分布的均值和标准差
    # pi1, pi2 = 0.5, 0.5    # 混合权重
    # data1 = np.random.normal(mu1, sigma1, size = (int(T*pi1), len(matrix)))  # 生成第一个分布的数据
    # data2 = np.random.normal(mu2, sigma2, size = (int(T*pi2), len(matrix)))  # 生成第二个分布的数据
    # noisy = np.concatenate([data1, data2])

    # gamma分布
    # noisy = np.random.gamma(1/10, 1, size=(T, num)) # 形状参数和尺度参数
    # noisy -= np.mean(noisy, axis=0)

    # 帕累托分布
    # noisy = (np.random.pareto(1, size=(T, num)) + 1) * 3
    # noisy -= np.mean(noisy, axis=0)
    
    # noisy = np.random.random(size = (T, len(matrix)))*np.sqrt(3)*2-np.sqrt(3)
    # noisy = np.random.random(size = (T, len(matrix)))*4 - 2
    
    # noisy = np.random.random(size = (T, len(matrix)))
    # noisy = np.where(noisy > 0.5, noisy+0.0, noisy-1.0)
    # noisy += 1
    # noisy = np.random.normal(0, 3, size = (T, len(matrix)))
    # noisy = noisy**3
    return noisy


# def get_noisy_reject(T, matrix):
#     num = len(matrix)
#     noisy = np.random.random(size = (T, len(matrix)))*3*2-3



def get_noisy_Gaussian(T, matrix):
    num = len(matrix)
    noisy = np.random.normal(0, 1, size = (T, len(matrix)))
    # for i in range(num):
    #     sigma = np.random.randint(1, 3)
    #     noisy[: , i] = np.random.normal(0, sigma, T)
    return noisy


def get_noisy_Mix(T, matrix, pos):
    noisy_nonG = get_noisy(T, matrix)
    noisy_G = get_noisy_Gaussian(T, matrix)
    nosiy = noisy_G
    for i in pos:
        # print("change: pos", i)
        nosiy[: ,i] = noisy_nonG[: ,i]
    return nosiy


# @jit(nopython=True)
def lower_triangle_graph(node, edge, max_indegree):
    matrix = np.zeros((node, node))
    visit = np.zeros(node)
    count = 0
    while count != edge:
        edge_set = np.random.randint(low=0, high=node, size=2)
        i = np.max(edge_set)
        j = np.min(edge_set)
        if i == j or matrix[i][j] or visit[i] >= max_indegree:
            continue
        else:
            visit[i] += 1
            matrix[i][j] = 1
            count += 1
  
    # return np.array(matrix)
    return matrix


# @jit(nopython=True)
def get_B_0(matrix):
    num = len(matrix)
    # B = np.random.random(size = (num, num))
    # B = np.where(B > 0.5, B, B - 1)
    # B = np.random.random(size = (num, num))
    B = np.random.random(size = (num, num))*0.5+0.5
    # B = np.where(B > 0.25, B+0.5, B - 1)

    B = np.where(matrix, B, 0)
    return B


# @jit(nopython=True)
def get_x(matrix, noisy, B_0):
    I = np.identity(len(B_0))
    x_1 = (np.linalg.pinv(I - B_0)).dot(noisy.T)
    return x_1


# @jit(nopython=True)
def generate_DAGs(node_num, edge_num, max_indegree, sample_size, dag_size):  

    DAG_list = list()
    B_list = list()
    noisy_list = list()
    data_list = list()
    
    for t in range(dag_size):
        DAG_temp = lower_triangle_graph(node_num, edge_num, max_indegree)
        DAG_list.append(DAG_temp)

        B_temp =  get_B_0(DAG_temp)
        B_list.append(B_temp)

        noisy_temp = get_noisy(sample_size, DAG_temp)
        noisy_list.append(noisy_temp)

        x_list_temp = get_x(DAG_temp, noisy_temp, B_temp).T
        data_list.append(x_list_temp)

    return DAG_list, data_list, B_list


def create_strictly_lower_triangular_matrix(n, num_elements):
    """生成一个n×n的严格下三角矩阵，其中恰好有num_elements个非零元素"""
    # 创建一个n×n的全零矩阵
    matrix = np.zeros((n, n))
    
    # 确定所有可能的下三角位置
    possible_positions = [(i, j) for i in range(1, n) for j in range(i)]
    
    # 确保非零元素数量不超过可能的位置数
    if num_elements > len(possible_positions):
        raise ValueError("The number of non-zero elements requested exceeds the maximum possible in a strictly lower triangular matrix.")
    
    # 随机选择指定数量的位置填充非零值
    fill_positions = np.random.choice(range(len(possible_positions)), size=num_elements, replace=False)
    
    # 填充这些位置
    for pos in fill_positions:
        i, j = possible_positions[pos]
        matrix[i, j] = 1  
    
    return matrix
