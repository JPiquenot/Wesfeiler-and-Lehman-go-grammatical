import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import (global_add_pool,global_mean_pool, global_max_pool)

mp =MessagePassing()

def edge_level_readout(x,C,batch_node,batch_edge,node_batch_edge,identite):
    I = torch.matmul(identite,C[0:1,:]*0+1)
    CI = C*I
    return torch.cat([CI,C-CI],1)


def node_level_readout_sum(x,C,batch_node,batch_edge,node_batch_edge,identite):
    I = torch.matmul(identite,C[0:1,:]*0+1)
    CI = C*I
    hI = global_add_pool(CI, node_batch_edge)
    hJ = global_add_pool(C-CI, node_batch_edge)
    return torch.cat([x,hI,hJ],1)

def node_level_readout_max(x,C,batch_node,batch_edge,node_batch_edge,identite):
    I = torch.matmul(identite,C[0:1,:]*0+1)
    CI = C*I
    hI = global_max_pool(CI, node_batch_edge)
    hJ = global_max_pool(C-CI, node_batch_edge)
    return torch.cat([x,hI,hJ],1)

def node_level_readout_mean(x,C,batch_node,batch_edge,node_batch_edge,identite):
    I = torch.matmul(identite,C[0:1,:]*0+1)
    CI = C*I
    hI = global_mean_pool(CI, node_batch_edge)
    hJ = global_mean_pool(C-CI, node_batch_edge)
    return torch.cat([x,hI,hJ],1)

def graph_level_readout_sum(x,C,batch_node,batch_edge,node_batch_edge,identite):
    I = torch.matmul(identite,C[0:1,:]*0+1)
    CI = C*I
    h = global_add_pool(x, batch_node)
    hI = global_add_pool(CI, batch_edge)
    hJ = global_add_pool(C-CI, batch_edge)
    return torch.cat([h,hI,hJ],1)

def graph_level_readout_max(x,C,batch_node,batch_edge,node_batch_edge,identite):
    I = torch.matmul(identite,C[0:1,:]*0+1)
    CI = C*I
    h = global_max_pool(x, batch_node)
    hI = global_max_pool(CI, batch_edge)
    hJ = global_max_pool(C-CI, batch_edge)
    return torch.cat([h,hI,hJ],1)

def graph_level_readout_mean(x,C,batch_node,batch_edge,node_batch_edge,identite):
    I = torch.matmul(identite,C[0:1,:]*0+1)
    CI = C*I
    h = global_mean_pool(x, batch_node)
    hI = global_mean_pool(CI, batch_edge)
    hJ = global_mean_pool(C-CI, batch_edge)
    return torch.cat([h,hI,hJ],1)

def edge_level_readout_PPGN(C,batch_node,batch_edge,node_batch_edge,identite):
    I = torch.matmul(identite,C[0:1,:]*0+1)
    CI = C*I
    return torch.cat([CI,C-CI],1)


def node_level_readout_sum_PPGN(C,batch_node,batch_edge,node_batch_edge,identite):
    I = torch.matmul(identite,C[0:1,:]*0+1)
    CI = C*I
    hI = global_add_pool(CI, node_batch_edge)
    hJ = global_add_pool(C-CI, node_batch_edge)
    return torch.cat([hI,hJ],1)

def node_level_readout_max_PPGN(C,batch_node,batch_edge,node_batch_edge,identite):
    I = torch.matmul(identite,C[0:1,:]*0+1)
    CI = C*I
    hI = global_max_pool(CI, node_batch_edge)
    hJ = global_max_pool(C-CI, node_batch_edge)
    return torch.cat([hI,hJ],1)

def node_level_readout_mean_PPGN(C,batch_node,batch_edge,node_batch_edge,identite):
    I = torch.matmul(identite,C[0:1,:]*0+1)
    CI = C*I
    hI = global_mean_pool(CI, node_batch_edge)
    hJ = global_mean_pool(C-CI, node_batch_edge)
    return torch.cat([hI,hJ],1)

def graph_level_readout_sum_PPGN(C,batch_node,batch_edge,node_batch_edge,identite):
    I = torch.matmul(identite,C[0:1,:]*0+1)
    CI = C*I
    # h = global_add_pool(x, batch_node)
    hI = global_add_pool(CI, batch_edge)
    hJ = global_add_pool(C-CI, batch_edge)
    return torch.cat([hI,hJ],1)

def graph_level_readout_max_PPGN(C,batch_node,batch_edge,node_batch_edge,identite):
    I = torch.matmul(identite,C[0:1,:]*0+1)
    CI = C*I
    # h = global_max_pool(x, batch_node)
    hI = global_max_pool(CI, batch_edge)
    hJ = global_max_pool(C-CI, batch_edge)
    return torch.cat([hI,hJ],1)

def graph_level_readout_mean_PPGN(C,batch_node,batch_edge,node_batch_edge,identite):
    I = torch.matmul(identite,C[0:1,:]*0+1)
    CI = C*I
    # h = global_mean_pool(x, batch_node)
    hI = global_mean_pool(CI, batch_edge)
    hJ = global_mean_pool(C-CI, batch_edge)
    return torch.cat([hI,hJ],1)