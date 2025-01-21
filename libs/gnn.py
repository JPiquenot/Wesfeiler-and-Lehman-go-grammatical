import torch
import torch.nn as nn
from libs.layer_gnn import G2N2Layer,PPGNLayer
from torch_geometric.nn import (global_add_pool,global_mean_pool, global_max_pool)
import libs.readout_gnn as ro

from time import time

def get_n_params(model):
    pp=0
    for p in model.parameters():
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class G2N2(nn.Module):
    def __init__(self,  node_input_dim, edge_input_dim, output_dim, device,
                 num_layer = 5, nodes_dim = [16,16,16,16,16], 
                 edges_dim = [16,16,16,16,16],decision_depth = 3,final_neuron = [512,256],
                 readout_type  = "sum" ,level = "graph",relu = True,dropout = .0):
        
        super(G2N2, self).__init__()
        
        self.num_layer = num_layer
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.output_dim = output_dim
        self.nodes_dim = nodes_dim
        self.edges_dim = edges_dim
        self.decision_depth = decision_depth
        self.final_neuron = final_neuron
        self.readout_type = readout_type
        self.level =level
        self.conv = nn.ModuleList()
        self.device = device
        self.relu = relu
        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(p = dropout)
         
        
        if self.num_layer < 1:
            raise ValueError("Number of GNN layer must be greater than 1")
            
        if self.num_layer != len(self.nodes_dim):
            raise ValueError("Number of GNN layer must match length of nodes_dim."+
                             "\n num_layer = {}, neuron_dim length = {}"
                             .format(self.num_layer,len(self.nodes_dim)))
            
        if self.num_layer != len(self.edges_dim):
            raise ValueError("Number of GNN layer must match length of neuron_dim."+
                             "\n num_layer = {}, neuron_dim length = {}"
                             .format(self.num_layer,len(self.edges_dim)))
        if self.decision_depth != len(self.final_neuron) + 1:
            raise ValueError("Number of decision layer must match in decision depth" + 
                             "={}, final neuron dim + 1 = {}".format(self.decision_depth,len(self.final_neuron) + 1))
        
        if self.level == "edge":
            self.readout = ro.edge_level_readout
        elif self.level == "node":
            if self.readout_type == "sum":
                self.readout = ro.node_level_readout_sum
            elif self.readout_type == "mean":
                self.readout = ro.node_level_readout_mean
            elif self.readout_type == "max":
                self.readout = ro.node_level_readout_max
            else:
                raise ValueError("Invalid readout type")
        elif self.level == "graph":
            if self.readout_type == "sum":
                self.readout = ro.graph_level_readout_sum
            elif self.readout_type == "mean":
                self.readout = ro.graph_level_readout_mean
            elif self.readout_type == "max":
                self.readout = ro.graph_level_readout_max
            else:
                raise ValueError("Invalid readout type")
        else:
            raise ValueError("Invalid level type, should be graph,node or edge")
        
        for i in range(self.num_layer):
            if i == 0:
                
                self.conv.append(G2N2Layer(nedgeinput= edge_input_dim, nedgeoutput = self.edges_dim[0],
                                            nnodeinput= self.node_input_dim, nnodeoutput= self.nodes_dim[0], device = self.device))

            else:
                self.conv.append(G2N2Layer(nedgeinput= self.edges_dim[i-1], nedgeoutput = self.edges_dim[i],
                                            nnodeinput= self.nodes_dim[i-1], nnodeoutput= self.nodes_dim[i], device = self.device))
        if self.level == "graph" or level == "node":
            self.fc = nn.ModuleList( [torch.nn.Linear(sum(self.nodes_dim)+2*sum(self.edges_dim)+node_input_dim + 2*edge_input_dim,self.final_neuron[0])])
            # node_out = node_input_dim + sum(nodes_dim) 
            # self.fc = nn.ModuleList( [torch.nn.Linear(node_out,self.final_neuron[0])])
        elif self.level == "edge":
            self.fc = nn.ModuleList( [torch.nn.Linear(2*self.edges_dim[-1],self.final_neuron[0])])
            
        else:
            raise ValueError("Invalid level type, should be graph,node or edge")
        
        for i in range(self.decision_depth-2):
            self.fc.append(torch.nn.Linear(self.final_neuron[i], self.final_neuron[i+1]))


        self.fc.append(torch.nn.Linear(self.final_neuron[-1], self.output_dim))
        
        
    
    def forward(self,data):
        x = data.x
        edge_index=data.edge_index3
        node_index = data.edge_index2
        C=data.edge_attr
        identite = C[:,0:1]
        batch_node = data.batch
        batch_edge = data.batch_edge
        num_node = data.num_node
        out_x = x
        out_C = C
        if self.relu:
            for i,l in enumerate(self.conv):
                x,C=(l(x, edge_index,node_index, C,batch_node,batch_edge,num_node))
                
                if i < self.num_layer - 1:
                    x = self.gelu(x)
                    C = self.gelu(C)
                x = self.dropout(x)
                C = self.dropout(C)
                out_x = torch.cat([out_x,x],1)
                out_C = torch.cat([out_C,C],1)
                
        else:
            for l in self.conv:
                x,C=(l(x, edge_index,node_index, C,batch_node,batch_edge,num_node))
                x = self.dropout(x)
                C = self.dropout(C)
                out_x = torch.cat([out_x,x],1)
                out_C = torch.cat([out_C,C],1)
        x = self.readout(out_x,out_C,data.batch,data.batch_edge,data.node_batch_edge,identite)
        # x= out_x
        for i in range(self.decision_depth-1):
            x = self.gelu(self.fc[i](x))
        return self.fc[-1](x) 
        
    
class PPGN(nn.Module):
    def __init__(self,  edge_input_dim, output_dim, device,
                 num_layer = 5, 
                 edges_dim = [16,16,16,16,16],decision_depth = 3,final_neuron = [512,256],
                 readout_type  = "sum" ,level = "graph",relu = True,dropout = 0):
        
        super(PPGN, self).__init__()
        
        self.num_layer = num_layer
        # self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.output_dim = output_dim
        # self.nodes_dim = nodes_dim
        self.edges_dim = edges_dim
        self.decision_depth = decision_depth
        self.final_neuron = final_neuron
        self.readout_type = readout_type
        self.level =level
        self.conv = nn.ModuleList()
        self.device = device
        self.relu = relu
        self.dropout = torch.nn.Dropout(p = dropout)
         
        
        if self.num_layer < 1:
            raise ValueError("Number of GNN layer must be greater than 1")
            
        # if self.num_layer != len(self.nodes_dim):
        #     raise ValueError("Number of GNN layer must match length of nodes_dim."+
        #                      "\n num_layer = {}, neuron_dim length = {}"
        #                      .format(self.num_layer,len(self.nodes_dim)))
            
        if self.num_layer != len(self.edges_dim):
            raise ValueError("Number of GNN layer must match length of neuron_dim."+
                             "\n num_layer = {}, neuron_dim length = {}"
                             .format(self.num_layer,len(self.edges_dim)))
        if self.decision_depth != len(self.final_neuron) + 1:
            raise ValueError("Number of decision layer must match in decision depth" + 
                             "={}, final neuron dim + 1 = {}".format(self.decision_depth,len(self.final_neuron) + 1))
        
        if self.level == "edge":
            self.readout = ro.edge_level_readout_PPGN
        elif self.level == "node":
            if self.readout_type == "sum":
                self.readout = ro.node_level_readout_sum_PPGN
            elif self.readout_type == "mean":
                self.readout = ro.node_level_readout_mean_PPGN
            elif self.readout_type == "max":
                self.readout = ro.node_level_readout_max_PPGN
            else:
                raise ValueError("Invalid readout type")
        elif self.level == "graph":
            if self.readout_type == "sum":
                self.readout = ro.graph_level_readout_sum_PPGN
            elif self.readout_type == "mean":
                self.readout = ro.graph_level_readout_mean_PPGN
            elif self.readout_type == "max":
                self.readout = ro.graph_level_readout_max_PPGN
            else:
                raise ValueError("Invalid readout type")
        else:
            raise ValueError("Invalid level type, should be graph,node or edge")
        
        for i in range(self.num_layer):
            if i == 0:
                
                self.conv.append(PPGNLayer(nedgeinput= edge_input_dim, nedgeoutput = self.edges_dim[0],
                                               device = self.device))

            else:
                self.conv.append(PPGNLayer(nedgeinput= self.edges_dim[i-1], nedgeoutput = self.edges_dim[i],
                                             device = self.device))
        if self.level == "graph" or level == "node":
            # self.fc = nn.ModuleList( [torch.nn.Linear(sum(self.nodes_dim)+2*sum(self.edges_dim)+node_input_dim + 2*edge_input_dim,self.final_neuron[0])])
            self.fc = nn.ModuleList( [torch.nn.Linear(sum(self.nodes_dim)+2*sum(self.edges_dim),self.final_neuron[0])])
        elif self.level == "edge":
            self.fc = nn.ModuleList( [torch.nn.Linear(2*self.edges_dim[-1],self.final_neuron[0])])
            
        else:
            raise ValueError("Invalid level type, should be graph,node or edge")
        
        for i in range(self.decision_depth-2):
            self.fc.append(torch.nn.Linear(self.final_neuron[i], self.final_neuron[i+1]))


        self.fc.append(torch.nn.LayerNorm(sum(self.nodes_dim)+2*sum(self.edges_dim), eps=1e-3,elementwise_affine=False))
        self.fc.append(torch.nn.Linear(self.final_neuron[-1], self.output_dim))
        self.norm = torch.nn.LayerNorm(edge_input_dim,eps=1e-3,elementwise_affine=False)
        
    
    def forward(self,data):
        x = data.x
        edge_index=data.edge_index2
        n = data.x.shape[0]//4
        A = data.edge_attr[2*n*n:3*n*n,1].reshape((n,n)).detach().cpu()
        B = data.edge_attr[n*n:2*n*n,1].reshape((n,n)).detach().cpu()
        torch.save(A,'data/A.npy')
        torch.save(B,'data/B.npy')
        # print((cs.path(A,2).sum()),(cs.path(B,2)).sum())
        # l,v = torch.linalg.eigh(torch.eye(18)*2. - A)
        # lb,vb = torch.linalg.eigh(torch.eye(18)*2. - B)
        # print((cs.path(A,6)*A).sum())
        # print((cs.path(B,6)*B).sum())
        # M = (A - A.mean())/A.std()
        # N = (B - B.mean())/B.std()
        # print((cs.path(M,3)*M).sum())
        # print((cs.path(N,3)*N).sum())
        # print(M)
        # for i in range(75):
        #     M = M@M
        #     N=N@N
        #     print(i)
        #     # print(torch.diag( M))
        #     # print(torch.diag( N))
        #     M = (M-M.mean())/M.std()
        #     N = (N-N.mean())/N.std()
        #     print(torch.diag( M).sum())
        #     print(torch.diag( N).sum())
        C=self.norm(data.edge_attr)
        identite = C[:,0:1]
        batch_node = data.batch
        res = []
        if self.relu:
            for j,l in enumerate(self.conv):
                x,C=(l(x, edge_index, C,batch_node))
                # print(x,C)
                res.append( self.readout(x,C,data.batch,data.batch_edge,data.node_batch_edge,identite))
        # x = self.readout(x,C,data.batch,data.batch_edge,data.node_batch_edge,identite)
                # for i in range(self.decision_depth-1):
                #     res = self.gelu(self.fc[j][i](res))
                # if j == 0:
                #     ret = self.fc[j][-1](res)
                # else:
                #     ret = self.fc[j][-2](ret + self.fc[j][-1](res))
                if j < self.num_layer - 1:
                    x = self.dropout(self.gelu(x))
                    C = self.dropout(self.gelu(C))
            res = self.fc[-2](torch.cat(res,dim=1))
            for i in range(self.decision_depth-1):
                res = self.gelu(self.fc[i](res))
            ret = self.fc[-1](res)
            
                
        else:
            for j,l in enumerate(self.conv):
                x,C=(l(x, edge_index, C,batch_node))
                # print(x,C)
                res.append( self.readout(x,C,data.batch,data.batch_edge,data.node_batch_edge,identite))
                x = self.dropout(x)
                C = self.dropou(C)
                # res = self.readout(x,C,data.batch,data.batch_edge,data.node_batch_edge,identite)
        # x = self.readout(x,C,data.batch,data.batch_edge,data.node_batch_edge,identite)
                # for i in range(self.decision_depth-1):
                #     res = self.gelu(self.fc[j][i](res))
                # if j == 0:
                #     ret = self.fc[j][-1](res)
                # else:
                #     ret = ret + self.fc[j][-1](res) 
            res = self.fc[-2](torch.cat(res,dim=1))
            for i in range(self.decision_depth-1):
                res = self.gelu(self.fc[i](res))
            ret = self.fc[-1](res)
        return ret