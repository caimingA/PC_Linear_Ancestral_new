import numpy as np
import causaldag as cd
from scipy import stats
import networkx as nx
import copy
from lingam.hsic import hsic_test_gamma

from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

import PC_LiNGAM_optimized as PL

from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils import Meek

# 模块级别的对象重用
_REG_OBJECT = LinearRegression(fit_intercept=False)


def matrix_to_edge(matrix):
    """将矩阵转换为边列表（优化版本）"""
    i_indices, j_indices = np.where(matrix)
    arcs = list(zip(j_indices, i_indices))
    return arcs


def edge_to_matrix(arcs, node_num):
    """将边列表转换为矩阵（向量化版本）"""
    matrix = np.zeros((node_num, node_num))
    if len(arcs) > 0:
        arcs_array = np.array(arcs)
        matrix[arcs_array[:, 1], arcs_array[:, 0]] = 1
    return matrix


def get_CPDAG(matrix):
    """获取CPDAG和colliders（优化版本）"""
    node_num = len(matrix)
    CPDAG = np.zeros((node_num, node_num))
    arcs = matrix_to_edge(matrix)
    g = cd.DAG(arcs=arcs)
    colliders = g.vstructures()
    cpdag = g.cpdag()
    edges = cpdag.edges
    direct_edges = cpdag.arcs
    
    # 处理有向边
    for j, i in direct_edges:
        CPDAG[i][j] = 1
        CPDAG[j][i] = -1
    
    # 处理无向边
    for j, i in edges:
        if CPDAG[i][j] == 0:
            CPDAG[i][j] = -1
            CPDAG[j][i] = -1
    
    return CPDAG, colliders


def get_descendants_of_all_colliders(CPDAG, collider_midnode_list):
    """获取所有collider中间节点的后代（优化版本）"""
    # 只保留有向边: CPDAG[i][j]==1 表示 i->j
    directed_matrix = (CPDAG == 1).astype(int)
    G = nx.from_numpy_array(directed_matrix.T, create_using=nx.DiGraph())
    
    except_set = set(collider_midnode_list)
    collider_descendants_dict = {}
    
    for mid_node in collider_midnode_list:
        descendants = nx.descendants(G, mid_node)
        collider_descendants_dict[mid_node] = list(descendants)
        except_set.update(descendants)
    
    return list(except_set), collider_descendants_dict


def get_descendants_of_one_node(G, node):
    """获取单个节点的后代"""
    descendants = list(nx.descendants(G, node))
    return descendants


def is_linear(X, Y, alpha):
    """判断是否线性相关"""
    _, p = stats.pearsonr(X, Y)
    return p < alpha


def get_residual(xi, xj):
    """计算残差"""
    return xi - (np.cov(xi, xj)[0, 1] / np.var(xj)) * xj


def is_independent(X, Y, alpha):
    """判断是否独立"""
    _, p = hsic_test_gamma(X, Y, bw_method="mdbs")
    return p > alpha


def is_Gaussian(X, alpha):
    """判断是否高斯分布"""
    _, p = stats.shapiro(X)
    return p > alpha


def get_unfinished_edges_nodes(CPDAG):
    """获取未完成的边和节点"""
    node_num = len(CPDAG)
    unfinished_edges = []
    finished_edges = []
    unfinished_nodes = set()
    all_nodes = set(range(node_num))
    
    for i in range(node_num):
        for j in range(i + 1, node_num):
            if CPDAG[i][j] == -1 and CPDAG[j][i] == -1:
                unfinished_edges.append([i, j])
                unfinished_nodes.update({i, j})
            else:
                finished_edges.append([i, j])
    
    finished_nodes = all_nodes - unfinished_nodes

    return (np.array(sorted(list(unfinished_nodes))), 
            unfinished_edges, 
            np.array(sorted(list(finished_nodes))), 
            finished_edges)


def do_Meek_rule(CPDAG):
    """应用Meek规则（优化版本）"""
    node_num = len(CPDAG)
    cg = CausalGraph(node_num)
    
    # 直接复制矩阵
    cg.G.graph = CPDAG.copy()
    
    cg = Meek.meek(cg)
    
    return cg.G.graph.astype(np.float64)


def get_True_CPDAG(DAG, change_list):
    """获取真实CPDAG"""
    node_num = len(DAG)
    CPDAG, colliders = get_CPDAG(DAG)

    unfinished_edge = []
    for i in range(node_num):
        for j in range(i+1, node_num):
            if CPDAG[i][j] == -1 and CPDAG[j][i] == -1:
                unfinished_edge.append([i, j])

    for i, j in unfinished_edge:
        if i in change_list or j in change_list:
            if DAG[i][j] == 1:
                CPDAG[i][j] = 1
            if DAG[j][i] == 1:
                CPDAG[j][i] = 1
            
    CPDAG = do_Meek_rule(CPDAG)

    return CPDAG


def evaluate(CPDAG, true_CPDAG):
    """评估CPDAG（优化版本）"""
    node_num = len(CPDAG)
    res_matrix = np.zeros((4, 4))
    
    # 定义边类型映射函数
    def get_edge_type(mat, i, j):
        if mat[i][j] == 0 and mat[j][i] == 0:
            return 0
        elif mat[i][j] == -1 and mat[j][i] == -1:
            return 1
        elif mat[i][j] == 1 and mat[j][i] == -1:
            return 2
        elif mat[i][j] == -1 and mat[j][i] == 1:
            return 3
        return 0
    
    for i in range(node_num):
        for j in range(i+1, node_num):
            cpdag_type = get_edge_type(CPDAG, i, j)
            true_type = get_edge_type(true_CPDAG, i, j)
            res_matrix[cpdag_type, true_type] += 1
    
    return res_matrix


def get_all_connected_componets(CPDAG, true_node_list):
    """获取所有连通分量（优化版本）"""
    # 创建邻接矩阵（无向）
    adj_matrix = (CPDAG != 0).astype(int)
    G = nx.from_numpy_array(adj_matrix, create_using=nx.Graph())
    
    connected_components = list(nx.connected_components(G))
    
    components_list = [
        [true_node_list[i] for i in sorted(comp)]
        for comp in connected_components
    ]
    
    return components_list


def get_all_roots(DAG, target):
    """获取所有根节点"""
    G = nx.from_numpy_array(DAG.T, create_using=nx.DiGraph())
    roots = [node for node in range(len(DAG)) if G.in_degree(node) == 0]
    reachable_roots = [root for root in roots if nx.has_path(G, root, target)]
    return reachable_roots


def extract_data_unfinished(data, CPDAG, unfinished_nodes, finished_nodes):
    """提取未完成节点的数据（优化版本）"""
    reg = _REG_OBJECT
    data_unfinished = []

    # 将CPDAG转换为布尔矩阵，便于快速检查父节点关系
    parent_relation = (CPDAG == 1) & (CPDAG.T == -1)

    for i in unfinished_nodes:
        # 找到所有与节点i相关的父节点
        parents_indices = np.where(parent_relation[i])[0]
        parents_indices = [j for j in parents_indices if j in finished_nodes]

        if len(parents_indices) != 0:
            parents_data = data[:, parents_indices]
            reg.fit(parents_data, data[:, i])
            residuals = data[:, i] - reg.predict(parents_data)
            data_unfinished.append(residuals)
        else:
            data_unfinished.append(data[:, i])

    return np.vstack(data_unfinished).T


def extract_data_unfinished_2(data, CPDAG, unfinished_nodes, finished_nodes):
    """提取未完成节点的数据（Lasso版本，已修复bug）"""
    parent_relation = (CPDAG == 1) & (CPDAG.T == -1)
    
    # 收集所有父节点
    all_parents = set()
    for i in unfinished_nodes:
        parents_indices = np.where(parent_relation[i])[0]
        parents_in_finished = [j for j in parents_indices if j in finished_nodes]
        all_parents.update(parents_in_finished)
    
    if len(all_parents) > 0:
        parents_list = sorted(list(all_parents))
        parents_data = data[:, parents_list]
        unfinished_data = data[:, unfinished_nodes]
        
        lasso = Lasso(0.1)
        lasso.fit(parents_data, unfinished_data)
        return unfinished_data - lasso.predict(parents_data)
    else:
        return data[:, unfinished_nodes]


def get_subCPDAG(CPDAG, node_list):
    """获取子CPDAG"""
    return CPDAG[np.ix_(node_list, node_list)]


def map_CPDAG_to_subCPDAG(X_data, CPDAG, components, true_nodes):
    """将CPDAG映射到子CPDAG（优化版本）"""
    mapping_dict = {node: idx for idx, node in enumerate(true_nodes)}
    mapping_list = []

    for comp in components:
        node_indices = [mapping_dict[node] for node in comp]
        sub_CPDAG = CPDAG[np.ix_(node_indices, node_indices)]
        sub_data = X_data[:, node_indices]
        mapping_list.append([comp, sub_CPDAG, sub_data])

    return mapping_list


def get_new_vstructure(matrix, colliders):
    """获取新的v-structure"""
    node_num = len(matrix)
    new_vstructures_list = []
    new_vstrcutures_edge_list = []
    
    for i in range(node_num):
        if i in colliders:
            continue
        
        direct_edge_list = []
        for j in range(node_num):
            if matrix[i][j] == 1 and matrix[j][i] == -1:
                for k in direct_edge_list:
                    if matrix[k][j] == 0:
                        if i not in new_vstructures_list:
                            new_vstructures_list.append(i)
                        if [k, i] not in new_vstrcutures_edge_list:
                            new_vstrcutures_edge_list.append([k, i])
                        if [j, i] not in new_vstrcutures_edge_list:
                            new_vstrcutures_edge_list.append([j, i])
                direct_edge_list.append(j)
    
    return new_vstructures_list, new_vstrcutures_edge_list


def detect_expections(CPDAG, collider_old):
    """检测异常"""
    node_num = len(CPDAG)
    DAG_temp = np.where(CPDAG == 1.0, 1.0, 0)
    loop_flag = PL.topological_sort(DAG_temp)
    
    if loop_flag == 0:
        return 1
    
    arcs = matrix_to_edge(DAG_temp)
    g = cd.DAG(arcs=arcs)
    collider_new = g.vstructures()
    
    for vs in collider_new:
        vs_inverse = (vs[2], vs[1], vs[0])
        if vs not in collider_old and vs_inverse not in collider_old:
            return 2
    
    return 0


def detect_exceptions_2(CPDAG, collider_old):
    """检测并修复异常（优化版本）"""
    node_num = len(CPDAG)
    DAG_temp = np.where(CPDAG == 1.0, 1.0, 0.0)
    topo_order = PL.topological_sort(DAG_temp)
    
    if len(topo_order) != node_num:
        unfinished_node = sorted(list(set(range(node_num)) - set(topo_order)))
        unfinished_num = len(unfinished_node)
        
        # 使用向量化操作
        for i in range(unfinished_num):
            for j in range(i+1, unfinished_num):
                ni, nj = unfinished_node[i], unfinished_node[j]
                if CPDAG[nj][ni] != 0.0:
                    CPDAG[nj][ni] = 1.0
                    CPDAG[ni][nj] = -1.0
    
    DAG_temp = np.where(CPDAG == 1.0, 1.0, 0.0)
    arcs = matrix_to_edge(DAG_temp)
    g = cd.DAG(arcs=arcs)
    collider_new = g.vstructures()
    
    for vs in collider_new:
        vs_inverse = (vs[2], vs[1], vs[0])
        if vs not in collider_old and vs_inverse not in collider_old:
            visit_list = []
            root_nodes_list = get_all_roots(DAG_temp, vs[1])
            
            if len(root_nodes_list) == 0:
                continue
            
            root_node = min(root_nodes_list)
            i_pos = root_node
            
            while i_pos is not None:
                next_list = []
                visit_list.append(i_pos)
                
                for j in range(node_num):
                    if CPDAG[j][i_pos] != 0 and j not in visit_list:
                        flag_temp = True
                        for vs_old in collider_old:
                            if (vs_old[0] == j and vs_old[1] == i_pos) or \
                               (vs_old[2] == j and vs_old[1] == i_pos):
                                flag_temp = False
                                break
                        
                        if flag_temp:
                            CPDAG[j][i_pos] = 1.0
                            CPDAG[i_pos][j] = -1.0
                            next_list.append(j)
                
                if len(next_list) != 0:
                    i_pos = min(next_list)
                else:
                    i_pos = None
    
    return CPDAG


def match_to_MEC(CPDAG, MEC, colliders):
    """匹配到MEC"""
    new_vstructures_list, new_vstrcutures_edge_list = get_new_vstructure(CPDAG, colliders)
    
    if len(new_vstructures_list) == 0:
        return CPDAG
    
    for c in new_vstrcutures_edge_list:
        i = c[1]
        j = c[0]
        CPDAG[j][i] = 1
        CPDAG[i][j] = -1
        nvl, nvel = get_new_vstructure(CPDAG, colliders)
        if len(nvel) >= len(new_vstrcutures_edge_list):
            CPDAG[j][i] = -1
            CPDAG[i][j] = 1
    
    return CPDAG
