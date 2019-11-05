import torch
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def build_adjacency(tensor_x, tensor_y, scaler):
    """ 根据欧氏距离构建邻接矩阵(实值、对称、有自环)
        Parameters
          tensor_x: pytorch tensor, with shape [m, d]
          tensor_y: pytorch tensor, with shape [n, d]
          scaler:  float, the scaler of adjust Euclidean distance
        Returns
          adjacency: pytorch tensor, with shape [m, n]
          相似度矩阵(邻接矩阵)
    """
    squared_sum_x = torch.pow(tensor_x, 2).sum(dim=1, keepdim=True)
    squared_sum_y = torch.pow(tensor_y, 2).sum(dim=1, keepdim=True)
    distances = squared_sum_x - 2 * torch.mm(tensor_x, tensor_y.t()) + squared_sum_y.t()
    adjacency = torch.exp(-1 * distances / scaler)
    return adjacency

def build_adjacency_topk(tensor_x, tensor_y, topk=10):
    """ 相似度前topk构建邻接矩阵(01值、非对称、不带自环)
        Parameters
          tensor_x: pytorch tensor, with shape [m, d]
          tensor_y: pytorch tensor, with shape [n, d]
          topk: 排序前topk的才有邻接关系
        Returns
          adjacency: pytorch tensor, with shape [m, n]
          相似度矩阵(邻接矩阵)
    """
    rbf_adjacency = rbf_kernel(tensor_x, tensor_y)
    topk_max = np.argsort(rbf_adjacency, axis=1)[:,-topk:]
    topk_adjacency = torch.zeros(rbf_adjacency.shape)
    if torch.cuda.is_available():
        topk_adjacency = topk_adjacency.cuda()
    for col_id in topk_max.T:
        topk_adjacency[np.arange(rbf_adjacency.shape[0]), col_id] = 1.0
    return topk_adjacency

def normalize_adj(adjacency, overflow_margin, device):
    """ 邻接矩阵规范化
        Parameters
          adjacency: pytorch tensor, with shape [m, n]
          带自环的邻接矩阵(否则需要adjacency + torch.eye(graph_size))
          overflow_margin: float
          防止孤立点(节点的度为0)溢出
        Returns 
          adjacency: pytorch tensor, with shape [m, n]
          根据度矩阵规范化后的邻接矩阵
    """
    graph_size = adjacency.shape[0]
    a = adjacency  # + torch.eye(graph_size)
    d = a @ torch.ones([graph_size, 1]).to(device)  # @是对tensor进行矩阵相乘
    d_inv_sqrt = torch.pow(d + overflow_margin, -0.5)
    d_inv_sqrt = torch.eye(graph_size).to(device) * d_inv_sqrt # *对tensor进行逐元素相乘
    return d_inv_sqrt @ a @ d_inv_sqrt


