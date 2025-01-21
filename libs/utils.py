import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.utils import to_undirected
import numpy as np
import scipy.io as sio
import libs.graphs as graph
import networkx as nx
import torch.nn.functional as Func
import pandas as pd
import libs.countsub as cs

import scipy.spatial.distance as dist




class GraphCountDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphCountDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["randomgraph.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A'][0]
        # list of output
        Y=a['F']

        data_list = []
        for i in range(len(A)):
            a=torch.Tensor(A[i])*1.
            cycle = []
            for j in range(3,8):
                cycle.append(cs.cycle(a,j).sum(1).unsqueeze(1)/2)
           
        
            # print(cycle)
            expy=torch.cat(cycle,1)

            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.ones(A[i].shape[0],1)
            edge_attr = None
            #y=torch.tensor(Y[i:i+1,:])            
            data_list.append(Data(edge_index=edge_index, x=x, y=expy, edge_attr = edge_attr))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


class HivDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(HivDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ["edge.csv.gz","edge-feat.csv.gz","graph-label.csv.gz","node-feat.csv.gz","num-edge-list.csv.gz","num-node-list.csv.gz"]

    @property
    def processed_file_names(self):
        return 'data.pt'


    def process(self):
        
        data_list =  []
        
        df_num_node = pd.read_csv(self.raw_paths[5], compression='gzip', header = None)
        df_num_edge = pd.read_csv(self.raw_paths[4], compression='gzip', header = None)
        df_node_feat = pd.read_csv(self.raw_paths[3], compression='gzip', header = None)
        df_y = pd.read_csv(self.raw_paths[2], compression='gzip', header = None)
        df_edge_feat = pd.read_csv(self.raw_paths[1], compression='gzip', header = None)
        df_edge = pd.read_csv(self.raw_paths[0], compression='gzip', header = None)
        
        loc_node = 0
        loc_edge = 0

        for i in range(len(df_num_node)):
            nod = np.array(df_num_node.iloc[[i]])[0][0]
            edg = np.array(df_num_edge.iloc[[i]])[0][0]
            E = np.array(df_edge.iloc[range(loc_edge,loc_edge+edg),0])
            F = np.array(df_edge.iloc[range(loc_edge,loc_edge+edg),1])
            y = torch.tensor(np.array(df_y.iloc[[i]])).type(torch.float32)
            edge_index = torch.Tensor(np.vstack((E,F))).type(torch.int64)
            x = torch.Tensor(np.array(df_node_feat.iloc[range(loc_node,loc_node+nod)])).type(torch.float32)
            
            edge_attr = torch.Tensor(np.array(df_edge_feat.iloc[range(loc_edge,loc_edge+edg)]))
            edge_attr = torch.cat([Func.one_hot(edge_attr[:,1].type(torch.int64),4).type(torch.float32),edge_attr[:,1:]],1)

    
            
            
                
            
    
                
            data_list.append(Data(edge_index=edge_index, x=x, y=y, edge_attr = edge_attr))
            
                
            loc_node += nod
            loc_edge += edg
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        
        torch.save((data, slices), self.processed_paths[0])

        
class TwoDGrid30(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TwoDGrid30, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["TwoDGrid30.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A']
        # list of output
        F=a['F']
        
        Y=a['Y'] 
        
        M = a['M']
        F=F.astype(np.float32)

        data_list = []
        for i in range(len(A)):
            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.reshape(torch.tensor(F[i]),(F[i].shape[0],1))
            # x = torch.cat([x,torch.ones(x.shape)],axis = 1)
            y=torch.tensor(Y[i])
            mask = torch.tensor(M[i])
            edge_attr = None
            data_list.append(Data(edge_index=edge_index, x=x, y=y,mask = mask, edge_attr=edge_attr))
        
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
class SRDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SRDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["sr251256.g6"]  #sr251256  sr351668

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]   
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for i,datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(),1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1,0))            
            data_list.append(Data(edge_index=edge_index, x=x, y=0))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


        
class G2N2design(object):   

    def __init__(self,operator = "adj",QM9 = False):
        
        
        # use laplacian or adjacency for spectrum
        self.operator = operator
        self.QM9 = QM9
        
     

    

    def __call__(self, data):
        if data.x is not None:
            n =data.x.shape[0]
        else:
            n = data.num_nodes
            data.x = torch.ones((n,1))
        data.x = data.x.type(torch.float)
        
        if data.edge_attr is not None:
            if len(data.edge_attr.shape)>1:
                nfeat = data.edge_attr.shape[1]
            else:
                nfeat = 1
                data.edge_attr = data.edge_attr.reshape((data.edge_attr.shape[0],1))

        
        if self.QM9:
            distance_mat = np.zeros((1,n,n))
            distance_mat[0,:,:] = dist.squareform(dist.pdist(data.pos))
        
        
        
               
        nsup=2
        
            
        A=np.zeros((n,n),dtype=np.float32)
        SP=np.zeros((nsup,n,n),dtype=np.float32) 
        A[data.edge_index[0],data.edge_index[1]]=1
        if np.linalg.norm(A-A.T)>0:
            A = A + A.T

        
    
        if self.operator == "lap":        
            A = graph.Laplaciannormal(A)
        if self.operator == "norm":
            A = graph.normalize(A)
            
        elif self.operator == "gcn":
            A = graph.gcnoperator(A)
        
        if self.operator == "cheb":
            A = graph.Laplaciannormal(A)
            V,U = np.linalg.eigh(A)
            vmax = V.max()
            A =  (2*A/vmax - np.eye(n))
                       
        
            SP[0,:,:] =  np.eye(n)
            SP[1,:,:] =  (2*A/vmax - np.eye(n))
                 
                          
            for i in range(2,nsup):
                SP[i,:,:]= (2*SP[2,:,:]@SP[i-1,:,:]-SP[i-2,:,:])
        else:
            for i in range(nsup):
                SP[i,:,:] = np.linalg.matrix_power(A,i)


        
        
        E=np.where(np.ones((n,n))>0)
        data.num_node = torch.zeros(1,dtype = torch.int64) + n
        data.batch_edge = torch.zeros(n*n,dtype=torch.int64)
        data.node_batch_edge = (torch.arange(n,dtype = torch.int64).reshape((n,1))@torch.ones((1,n),dtype = torch.int64)).reshape(n*n)
        data.edge_index2=torch.cat([torch.zeros((1,n),dtype=torch.int64),torch.arange(n,dtype = torch.int64).unsqueeze(0)],0)
        data.edge_index3=torch.Tensor(np.vstack((data.batch_edge,E[0],E[1]))).type(torch.int64)
        if data.edge_attr is not None:
            C = np.zeros((nfeat,n,n))
            for i in range(nfeat):
                C[i,data.edge_index[0],data.edge_index[1]] = data.edge_attr[:,i]
                res = C[i,:,:]
                if np.linalg.norm(res- res.T)>0:
                    res = res+ res.T
                if self.operator == 'norm':
                    res = graph.normalize(res)
                C[i,:,:] = res
            if self.QM9:
                data.edge_attr = torch.cat([torch.Tensor(SP[:,E[0],E[1]].T).type(torch.float32),torch.Tensor(C[:,E[0],E[1]].T).type(torch.float32),torch.Tensor(distance_mat[:,E[0],E[1]].T).type(torch.float32)],1)
            else:
                data.edge_attr = torch.cat([torch.Tensor(SP[:,E[0],E[1]].T).type(torch.float32),torch.Tensor(C[:,E[0],E[1]].T).type(torch.float32)],1)
        else:
            data.edge_attr = torch.Tensor(SP[:,E[0],E[1]].T).type(torch.float32)
                
        return data
    
class PPGNdesign(object):   

    def __init__(self,operator = "adj",QM9 = False):
        
        
        # use laplacian or adjacency for spectrum
        self.operator = operator
        self.QM9 = QM9
        
     

    

    def __call__(self, data):
        if data.x is not None:
            n =data.x.shape[0]
        else:
            n = data.num_nodes
            data.x = torch.ones((n,1))
        data.x = data.x.type(torch.float)
        n_node_feat = data.x.shape[1]
        
        if data.edge_attr is not None:
            if len(data.edge_attr.shape)>1:
                nfeat = data.edge_attr.shape[1]
            else:
                nfeat = 1
                data.edge_attr = data.edge_attr.reshape((data.edge_attr.shape[0],1))

        
        if self.QM9:
            distance_mat = np.zeros((1,n,n))
            distance_mat[0,:,:] = dist.squareform(dist.pdist(data.pos))
        
        
        
               
        nsup=2
        
            
        A=np.zeros((n,n),dtype=np.float32)
        SP=np.zeros((nsup,n,n),dtype=np.float32) 
        A[data.edge_index[0],data.edge_index[1]]=1
        if np.linalg.norm(A-A.T)>0:
            A = A + A.T

        
    
        if self.operator == "lap":        
            A = graph.Laplaciannormal(A)
        if self.operator == "norm":
            A = graph.normalize(A)
            
        elif self.operator == "gcn":
            A = graph.gcnoperator(A)
        
        if self.operator == "cheb":
            A = graph.Laplaciannormal(A)
            V,U = np.linalg.eigh(A)
            vmax = V.max()
            A =  (2*A/vmax - np.eye(n))
                       
        
            SP[0,:,:] =  np.eye(n)
            SP[1,:,:] =  (2*A/vmax - np.eye(n))
                 
                          
            for i in range(2,nsup):
                SP[i,:,:]= (2*SP[2,:,:]@SP[i-1,:,:]-SP[i-2,:,:])
        else:
            for i in range(nsup):
                SP[i,:,:] = np.linalg.matrix_power(A,i)


        
        I = np.where(np.eye(n)>0)
        E=np.where(np.ones((n,n))>0)

        data.batch_edge = torch.zeros(n*n,dtype=torch.int64)
        data.node_batch_edge = (torch.arange(n,dtype = torch.int64).reshape((n,1))@torch.ones((1,n),dtype = torch.int64)).reshape(n*n)
        data.edge_index2=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
        data.edge_index3=torch.Tensor(np.vstack((data.batch_edge,E[0],E[1]))).type(torch.int64)
        if data.edge_attr is not None:
            C = np.zeros((nfeat+n_node_feat,n,n))
            for i in range(nfeat):
                C[i,data.edge_index[0],data.edge_index[1]] = data.edge_attr[:,i]
                res = C[i,:,:]
                if np.linalg.norm(res- res.T)>0:
                    res = res+ res.T
                if self.operator == 'norm':
                    res = graph.normalize(res)
                C[i,:,:] = res
            for i in range(n_node_feat):
                C[i+nfeat,I[0],I[1]] = data.x[:,i]
            if self.QM9:
                data.edge_attr = torch.cat([torch.Tensor(SP[:,E[0],E[1]].T).type(torch.float32),torch.Tensor(C[:,E[0],E[1]].T).type(torch.float32),torch.Tensor(distance_mat[:,E[0],E[1]].T).type(torch.float32)],1)
            else:
                data.edge_attr = torch.cat([torch.Tensor(SP[:,E[0],E[1]].T).type(torch.float32),torch.Tensor(C[:,E[0],E[1]].T).type(torch.float32)],1)
        else:
            data.edge_attr = torch.Tensor(SP[:,E[0],E[1]].T).type(torch.float32)
                
        return data
    