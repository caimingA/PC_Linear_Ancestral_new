import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import lingam

def draw_undirected_graph(adjacency_matrix, name_list = []):
    matrix = adjacency_matrix.T
    length = 8
    node_num = len(matrix)
    
    pos_list = list()
    # pos_list = [(10, 10), (5, 7), (10, 7), (2, 4), (4, 4), (6, 4), (8, 4), (1, 1), (2, 1), (3, 1)]
    y_gap = (length - 1) / np.log2(node_num)

    # count_layer = 0
    count_num = 0
    for i in range(int(np.log2(node_num)) + 1):
        if 2**i > node_num - count_num:
            x_gap = (length - 1) / (node_num - count_num + 1)
            for j in range(node_num - count_num):
                pos_list.append((0.5 + x_gap * (j + 1), length - 0.5 - y_gap*i - y_gap / (node_num - count_num)))
                count_num += 1
        else:
            x_gap = (length - 1) / (2**i + 1)
            for j in range(2**i):
                if count_num == node_num:
                    break
                pos_list.append((0.5 + x_gap * (j + 1), length - 0.5 - y_gap*i - y_gap / 2**i))
                count_num += 1
    temp = nx.Graph()
    G = nx.from_numpy_array(matrix, create_using=temp)
    nx.draw(G, pos=pos_list, with_labels=True, node_color='lightblue', edge_color='gray')

    plt.show()


def draw_directed_graph(adjacency_matrix, change_list = []):
    matrix = adjacency_matrix.T
    length = 8
    node_num = len(matrix)
    node_list = [i for i in range(node_num)]
    
    pos_list = list()
    # pos_list = [(10, 10), (5, 7), (10, 7), (2, 4), (4, 4), (6, 4), (8, 4), (1, 1), (2, 1), (3, 1)]
    # node_shape_list = list()
    # for i in range(node_num):
    #     if i in change_list:
    #         node_shape_list.append('^')
    #     else:
    #         node_shape_list.append('o')
    y_gap = (length - 1) / np.log2(node_num)

    # count_layer = 0
    count_num = 0
    for i in range(int(np.log2(node_num)) + 1):
        if 2**i > node_num - count_num:
            x_gap = (length - 1) / (node_num - count_num + 1)
            for j in range(node_num - count_num):
                # coef = j if j // 2 == 0 else -j
                coef = 1
                pos_list.append((0.5 + x_gap * (j + 1), length - 0.5 - y_gap*i - coef*(y_gap / (node_num - count_num))))
                # print(length - 0.5 - y_gap*i)
                count_num += 1
        else:
            x_gap = (length - 1) / (2**i + 1)
            for j in range(2**i):
                # coef = j if j // 2 == 0 else -j
                coef = 1
                if count_num == node_num:
                    break
                pos_list.append((0.5 + x_gap * (j + 1), length - 0.5 - y_gap*i - coef*(y_gap / 2**i)))
                count_num += 1
    
    # print(pos_list)
    temp = nx.DiGraph()
    diamond_nodes = change_list
    circle_nodes = list(set(node_list) - set(diamond_nodes))
    G = nx.from_numpy_array(matrix, create_using=temp)
    nx.draw(G, pos=pos_list, with_labels=True, nodelist=circle_nodes, node_color='lightblue', edge_color='gray', node_shape='o')
    nx.draw(G, pos=pos_list, with_labels=True, nodelist=diamond_nodes, node_color='lightblue', edge_color='gray', node_shape='d')
    # nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
    # print(G.edges())
    plt.show()


def draw_CPDAG(adjacency_matrix, change_list = []):
    # print(adjacency_matrix)
    matrix = adjacency_matrix
    length = 8
    node_num = len(matrix)
    node_list = [i for i in range(node_num)]
    diamond_nodes = change_list
    circle_nodes = list(set(node_list) - set(diamond_nodes))

    pos_list = list()
    # pos_list = [(10, 10), (5, 7), (10, 7), (2, 4), (4, 4), (6, 4), (8, 4), (1, 1), (2, 1), (3, 1)]
    y_gap = (length - 1) / np.log2(node_num)

    # node_shape_list = list()
    # for i in range(node_num):
    #     if i in change_list:
    #         node_shape_list.append('^')
    #     else:
    #         node_shape_list.append('o')
    # count_layer = 0
    count_num = 0
    for i in range(int(np.log2(node_num)) + 1):
        if 2**i > node_num - count_num:
            x_gap = (length - 1) / (node_num - count_num + 1)
            for j in range(node_num - count_num):
                pos_list.append((0.5 + x_gap * (j + 1), length - 0.5 - y_gap*i - y_gap / (node_num - count_num)))
                count_num += 1
        else:
            x_gap = (length - 1) / (2**i + 1)
            for j in range(2**i):
                if count_num == node_num:
                    break
                pos_list.append((0.5 + x_gap * (j + 1), length - 0.5 - y_gap*i - y_gap / 2**i))
                count_num += 1
    
    
    G = nx.DiGraph()

    for i in range(node_num):
        G.add_node(i)
        # if i in change_list:
        #     G.add_node(i, node_shpae = 'd')
        # else:
        #     G.add_node(i, node_shpae = 'o')

    # pos_list = nx.planar_layout(G)

    nx.draw(
        G
        , with_labels=True
        , pos=pos_list
        , node_color='lightblue'
        # , nodelist=circle_nodes
        # , node_shpae = 'o'
        )
    
    # nx.draw(
    #     G
    #     , with_labels=True
    #     , pos=pos_list
    #     , node_color='lightblue'
    #     , nodelist=diamond_nodes
    #     , node_shpae = 'd'
    #     )
    
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if(matrix[j][i] == 1 and matrix[i][j] == -1): # 后 -> 前
                G.add_edge(i, j, arr='->')
            if(matrix[j][i] == -1 and matrix[i][j] == -1):
                ind_max = np.max([i, j])
                ind_min = np.min([i, j])
                G.add_edge(ind_max, ind_min, arr='-')
            if(matrix[j][i] == 1 and matrix[i][j] == 1):
                ind_max = np.max([i, j])
                ind_min = np.min([i, j])
                G.add_edge(ind_max, ind_min, arr='<->')
    
    edges = G.edges()

    arrs = [G[u][v]['arr'] for u,v in edges]
    edge_list = list(edges)

    
    
    for i in range(len(edge_list)):
        nx.draw_networkx_edges(
            G
            , edgelist=[edge_list[i]]
            , pos=pos_list
            , arrowstyle=arrs[i]
            , edge_color='gray'
            )
    
    # G = nx.from_numpy_array(matrix, create_using=temp)
    # nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')

    plt.show()