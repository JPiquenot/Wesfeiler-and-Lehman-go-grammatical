from torch_geometric.typing import OptTensor
import math
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from time import time




def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)





class Conv_agg(torch.nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, device, K=1,bias=True):
        super(Conv_agg, self).__init__()

        assert K > 0       
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shapetensor = torch.zeros((K,1)).to(device)
        self.device = device

        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))
       
        if bias:
           self.bias = Parameter(torch.Tensor(out_channels))
        else:
           self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)


    def forward(self, h, X,edge_index,node_index,batch_node,batch_edge,num_node):
        """"""
        # print(h.shape, X.shape)
        N = num_node.max().item()
        # print(N)
        test = torch.zeros(edge_index.shape,dtype = torch.int64)
        test[0,:] = batch_edge
        test[1,:] = edge_index[1,:] - edge_index[0,:]
        test[2,:] = edge_index[2,:] - edge_index[0,:]
        # print(node_index.shape,batch_node.shape)
        node_t = torch.zeros(node_index.shape,dtype = torch.int64) 
        node_t[0,:] = batch_node
        node_t[1,:] = node_index[1,:]-node_index[0,:]
        # print(batch_node,node_index[1,:]-node_index[0,:])

        
        resx = torch.zeros((X.shape[1],num_node.shape[0],N,N),device = self.device)
        resx[:,test[0],test[1],test[2]] = X.T
        resh = torch.zeros((num_node.shape[0],N,h.shape[1]),device = self.device)
        resh[node_t[0],node_t[1],:] = h
        
        res = torch.matmul(resx,resh)
        # print(res.shape,self.weight.shape)
        # res =(res- res.mean(2,keepdim = True))/(res.mean(2,keepdim = True) + 1e-5)
        res = res[:,node_t[0],node_t[1],:]
        # print(res.shape,self.weight.shape)
        res = torch.matmul(res,self.weight).sum(0)    
        # print(h.shape,res.shape)
        
        if self.bias is not None:
            res += self.bias
        
        return res
    
    
    def __repr__(self):
        return '{}({}, {}, K={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,self.weight.size(0))


class G2N2Layer(torch.nn.Module):
    
    def __init__(self, nedgeinput,nedgeoutput,nnodeinput,nnodeoutput,device):
        super(G2N2Layer, self).__init__()

        self.nedgeinput = nedgeinput
        self.nnodeinput = nnodeinput
        self.nnodeoutput = nnodeoutput
        self.shapetensor = torch.zeros(nedgeinput,1).to(device)
        self.device = device
         
        
        

        self.L1 = torch.nn.Linear(nedgeinput, nedgeinput,bias=False)
        self.L2 = torch.nn.Linear(nedgeinput, nedgeinput,bias=False)
        self.L3 = torch.nn.Linear(nedgeinput, nedgeinput,bias=False)
        self.L4 = torch.nn.Linear(nedgeinput, nedgeinput,bias=False)
        self.L5 = torch.nn.Linear(nnodeinput,max(nnodeinput,nedgeinput),bias = False)
        
        
        
        self.mlp_edge = torch.nn.Sequential(
                # torch.nn.LayerNorm(3*nedgeinput + max(nnodeinput,nedgeinput),eps = 1e-5,elementwise_affine=True),
                # torch.nn.BatchNorm1d(3*nedgeinput + max(nnodeinput,nedgeinput),eps = 1e-5,track_running_stats=False),
                torch.nn.Linear(3*nedgeinput + max(nnodeinput,nedgeinput)  ,4*nedgeinput,bias=False),
                # torch.nn.Linear(3*nedgeinput + max(nnodeinput,nedgeinput)  ,nedgeoutput,bias=False),
                torch.nn.LayerNorm(4*nedgeinput,eps = 1e-5,elementwise_affine=True),
                # torch.nn.LayerNorm(nedgeoutput,eps = 1e-5,elementwise_affine=False),
                torch.nn.GELU(),
                torch.nn.Linear(4*nedgeinput ,nedgeoutput,bias=False),
                # torch.nn.LayerNorm(nedgeoutput,eps = 1e-5,elementwise_affine=False)
                # torch.nn.BatchNorm1d(nedgeoutput ,eps = 1e-5,track_running_stats=False)
                )
        
        self.mlp_node = torch.nn.Sequential(
                # torch.nn.LayerNorm(nnodeoutput,eps = 1e-5,elementwise_affine=True),
                # torch.nn.BatchNorm1d(nnodeoutput,eps = 1e-5,track_running_stats=False),
                torch.nn.Linear(nnodeinput ,4*nnodeinput,bias=False),
                torch.nn.LayerNorm(4*nnodeinput,eps = 1e-5,elementwise_affine=True),
                # torch.nn.Linear(nnodeinput ,nnodeoutput,bias=False),
                # torch.nn.LayerNorm(nnodeoutput,eps = 1e-5,elementwise_affine=False),
                torch.nn.GELU(),
                torch.nn.Linear(4*nnodeinput ,nnodeoutput,bias=False),
                # torch.nn.LayerNorm(nnodeoutput,eps = 1e-5,elementwise_affine=False)
                # torch.nn.BatchNorm1d(nnodeoutput,eps = 1e-5,track_running_stats=False),
                )
            
        
        self.agg = Conv_agg(nnodeinput,nnodeinput, device, K=nedgeoutput,bias = False)
        # self.norm1 = torch.nn.BatchNorm1d(nnodeoutput,eps = 1e-5,track_running_stats=False)
        # self.norm1 = torch.nn.LayerNorm(nnodeoutput,eps = 1e-5,elementwise_affine=True)
        # self.normhadam = torch.nn.BatchNorm1d(nedgeinput,eps = 1e-5,track_running_stats=False)
        # self.normmatmul = torch.nn.BatchNorm1d(nedgeinput,eps = 1e-5,track_running_stats=False)
        # 
        # self.norm2 = torch.nn.LayerNorm(nedgeoutput,eps = 1e-5,elementwise_affine=True)

    
     
    
    def matmul(self,X,Y,batch_node,edge_index):
        
        zer = torch.unsqueeze(batch_node*0.,0)

        
        resx = torch.matmul(torch.unsqueeze(torch.matmul(self.shapetensor,zer),2),zer)

        resy = torch.matmul(torch.unsqueeze(torch.matmul(self.shapetensor,zer),2),zer)
       
       
        resx[:,edge_index[0],edge_index[1]] = X.T
        
        resy[:,edge_index[0],edge_index[1]] = Y.T
        
        res = torch.matmul(resx,resy)
        
        return res[:,edge_index[0],edge_index[1]].T
    
    def matmul2(self,X,Y,batch_node,edge_index,num_node):
        
        
        
        N = num_node.max().item()
        test = torch.zeros(edge_index.shape,dtype = torch.int64)
        test[0,:] = batch_node
        test[1,:] = edge_index[1,:] - edge_index[0,:]
        test[2,:] = edge_index[2,:] - edge_index[0,:]

        
        resx = torch.zeros((num_node.shape[0],X.shape[1],N,N),device = self.device)

        resy = torch.zeros((num_node.shape[0],X.shape[1],N,N),device = self.device)
       
       
        resx[test[0],:,test[1],test[2]] = X
        
        resy[test[0],:,test[1],test[2]] = Y
        
        res = torch.matmul(resx,resy)
        # res =(res- res.mean((2,3),keepdim = True))/(res.mean((2,3),keepdim = True) + 1e-5)
        return res[test[0],:,test[1],test[2]]
    
       
    
    def diag(self,h,edge_index):
        res2= torch.diag_embed(h.T)
        return   res2[:,edge_index[1],edge_index[2]].T
            
    

    def forward(self, x,edge_index,node_index,C,batch_node,batch_edge,num_node):
        
        
        
        tmp_diag = self.diag( (self.L5(x)),edge_index)
        tmp_matmul = self.matmul2(  (self.L3(C)),  (self.L4(C)),batch_edge, edge_index,num_node)
        tmp=torch.cat([  (C),(self.L1(C))*  (self.L2(C)),tmp_diag,tmp_matmul],1)
        Cout = self.mlp_edge(tmp)
        
        # xout=(self.agg(x, Cout, edge_index, batch_node))
        
        
        # tmp_diag = self.diag( (self.L5(x)/self.nnodeinput),edge_index)
        # tmp_matmul = self.matmul(  (self.L3(C)/self.nedgeinput),  (self.L4(C)/self.nedgeinput),batch_node, edge_index)
        # tmp=torch.cat([  (C),tmp_matmul],1)
        # Cout = self.mlp2(torch.relu((self.mlp1(tmp))))
        
        xout= self.mlp_node((self.agg(x, Cout, edge_index,node_index, batch_node,batch_edge,num_node)))
        # N = num_node.max().item()
        # # print(N)
        # test = torch.zeros(edge_index.shape,dtype = torch.int64)
        # test[0,:] = batch_edge
        # test[1,:] = edge_index[1,:] - edge_index[0,:]
        # test[2,:] = edge_index[2,:] - edge_index[0,:]
        # # print(node_index.shape,batch_node.shape)
        # node_t = torch.zeros(node_index.shape,dtype = torch.int64) 
        # node_t[0,:] = batch_node
        # node_t[1,:] = node_index[1,:]-node_index[0,:]
        # # print(batch_node,node_index[1,:]-node_index[0,:])

        
        # res = torch.zeros((Cout.shape[1],num_node.shape[0],N,N),device = self.device)
        # res[:,test[0],test[1],test[2]] = Cout.T
        # resh = torch.zeros((num_node.shape[0],N,xout.shape[1]),device = self.device)
        # resh[node_t[0],node_t[1],:] = xout
        # res =(res- res.mean((2,3),keepdim = True))/(res.mean((2,3),keepdim = True) + 1e-5)
        # resh =(resh- resh.mean((2),keepdim = True))/(resh.mean((2),keepdim = True) + 1e-5)
        # return self.norm1(xout) ,  self.norm2(Cout)
        return xout ,  Cout
    
class PPGNLayer(torch.nn.Module):
    
    def __init__(self, nedgeinput,nedgeoutput,device):
        super(PPGNLayer, self).__init__()

        self.nedgeinput = nedgeinput
        # self.nnodeinput = nnodeinput
        # self.nnodeoutput = nnodeoutput
        self.shapetensor = torch.zeros(nedgeinput,1).to(device)
        self.device = device
         
        
        

        self.L1 = torch.nn.Linear(nedgeinput, nedgeinput,bias=False)
        self.L2 = torch.nn.Linear(nedgeinput, nedgeinput,bias=False)
        
        
        
        
        self.mlp1 = torch.nn.Linear(2*nedgeinput ,8*nedgeinput,bias=False)
        self.mlp2 = torch.nn.Linear(8*nedgeinput ,nedgeoutput,bias=False)
        
        # self.mlp3 = torch.nn.Linear(nnodeinput ,8*nnodeinput,bias=False)
        # self.mlp4 = torch.nn.Linear(8*nnodeinput ,nnodeoutput,bias=False)
            
        
        # self.agg = Conv_agg(nnodeinput,nnodeinput, device, K=nedgeoutput,bias = True)

    
     
    
    def matmul(self,X,Y,batch_node,edge_index):
        
        zer = torch.unsqueeze(batch_node*0.,0)

        
        resx = torch.matmul(torch.unsqueeze(torch.matmul(self.shapetensor,zer),2),zer)

        resy = torch.matmul(torch.unsqueeze(torch.matmul(self.shapetensor,zer),2),zer)
       
       
        resx[:,edge_index[0],edge_index[1]] = X.T
        
        resy[:,edge_index[0],edge_index[1]] = Y.T
        
        res = torch.matmul(resx,resy)
        
        return res[:,edge_index[0],edge_index[1]].T
    
    
    
       
    
    def diag(self,h,edge_index):
        res2= torch.diag_embed(h.T)
        return   res2[:,edge_index[0],edge_index[1]].T
            
    

    def forward(self,edge_index,C,batch_node):
        
        
        # tmp_diag = self.diag( (self.L5(x)),edge_index)
        tmp_matmul = self.matmul(  (self.L1(C)),  (self.L2(C)),batch_node, edge_index)
        tmp=torch.cat([  (C),tmp_matmul],1)
        Cout = self.mlp2(torch.relu((self.mlp1(tmp))))
        
        # xout=(self.agg(x, Cout, edge_index, batch_node))
        
        
        # tmp_diag = self.diag( (self.L5(x)/self.nnodeinput),edge_index)
        # tmp_matmul = self.matmul(  (self.L3(C)/self.nedgeinput),  (self.L4(C)/self.nedgeinput),batch_node, edge_index)
        # tmp=torch.cat([  (C),tmp_matmul],1)
        # Cout = self.mlp2(torch.relu((self.mlp1(tmp))))
        
        # xout=self.mlp4(torch.relu(self.mlp3((self.agg(x, Cout, edge_index, batch_node)))))
        
        return  Cout
    

