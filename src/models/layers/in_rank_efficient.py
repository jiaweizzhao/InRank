import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init

@torch.jit.script
def _jit_low_rank_matrix_mul(current_modes:int, buffer_modes:int, x, A, B, bias):
    return F.linear(F.linear(x, A[:current_modes+buffer_modes,:], None), B[:,:current_modes+buffer_modes], bias)

@torch.jit.script
def _jit_compute_explained_variance(s_max:int, s):
    s_current = s.clone()
    s_current[s_max:] = 0
    return 1 - torch.var(s - s_current) / torch.var(s)
    
class InRankEfficient(nn.Module):
    '''
    updated version supports dynamic low-rank matrix multiplication
    '''
    def __init__(self, in_features, out_features, *args, bias=True, init_scale = 1.0, **kwargs):
        super().__init__()
        
        max_rank = int(min(in_features, out_features)/2)
        self.A = nn.Parameter(torch.empty((max_rank, in_features)))
        self.B = nn.Parameter(torch.empty((out_features, max_rank)))
        init.kaiming_uniform_(self.A, a=math.sqrt(5))
        init.kaiming_uniform_(self.B, a=math.sqrt(5))
        # scaling
        self.A.data.mul_(init_scale)
        self.B.data.mul_(init_scale)
        
        self.bias_flag = bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            self.bias.data.zero_()
        else:
            self.register_parameter('bias', None)
        
        # incremental configs
        self.init_modes = kwargs.get('init_modes')
        self.buffer_modes = kwargs.get('buffer_modes')
        self.explained_ratio_threshold = kwargs.get('explained_ratio_threshold')
        self.warmup_iter = kwargs.get('warmup_iter')
        self.stage = kwargs.get('stage')
        self.gap_iter = kwargs.get('gap_iter')
        
        self.max_modes = max_rank
        
        self.current_explained_ratio = 0
        self.current_s_vector = None
        
        # support resume from checkpoint
        self.current_modes_tensor = nn.Parameter(torch.tensor([self.init_modes], dtype = torch.float32))
        self.iter_tensor = nn.Parameter(torch.tensor([0], dtype = torch.float32))
        self.current_modes_tensor.requires_grad = False
        self.iter_tensor.requires_grad = False
        
        self.current_modes = int(self.current_modes_tensor.data.item())
        self.iter = int(self.iter_tensor.data.item())
        
    def forward(self, x):
        self.current_modes = int(self.current_modes_tensor.data.item())
        self.iter = int(self.iter_tensor.data.item())
        
        self.iter += 1
        self.iter_tensor.data = torch.tensor([self.iter], dtype = torch.float32) # keep self.iter_tensor in sync with self.iter
        
        if self.stage == 'stage_2':
            return _jit_low_rank_matrix_mul(self.current_modes, 0, x, self.A, self.B, self.bias)
        
        if self.iter > self.warmup_iter and self.iter % self.gap_iter == 0:
            self._rank_determination()
            self.current_modes_tensor.data = torch.tensor([self.current_modes], dtype = torch.float32) # keep self.current_modes_tensor in sync with self.current_modes
        
        return _jit_low_rank_matrix_mul(self.current_modes, self.buffer_modes, x, self.A, self.B, self.bias)
    
    def _rank_determination(self):
        '''
        determine the rank of the matrix based on incremental behavior
        '''
        
        # SVD
        matrix = torch.matmul(self.A.data[:self.current_modes+self.buffer_modes,:].T, self.B.data[:,:self.current_modes+self.buffer_modes].T)
        if matrix.dtype != torch.float:
            matrix = matrix.float()
        else:
            matrix = matrix
        s= torch.linalg.svdvals(matrix)
        
        assert self.current_modes <= self.max_modes - self.buffer_modes
        if self.current_modes < self.max_modes - self.buffer_modes:
            current_modes, ratio = self._matrx_svd_approximation_explained_variance(s, self.explained_ratio_threshold, self.current_modes, self.buffer_modes, self.max_modes)
            self.current_modes = current_modes
            self.current_explained_ratio = ratio
            self.current_s_vector = s
        
    def _matrx_svd_approximation_explained_variance(self, s, explained_var_ratio, min_rank, buffer_rank, max_rank):
        """
        approximate matrix by SVD and compute how many ranks are needed to explain the matrix above a threshold
        
        Note: produce current_modes not larger than max_modes - buffer_modes
        """
        for index in range(min_rank, min(min_rank + buffer_rank, max_rank - buffer_rank)):
            assert index <= self.max_modes
            ratio = _jit_compute_explained_variance(index, s)
            if ratio > explained_var_ratio:
                break
        
        return index, ratio
    
    def resize(self, if_starting, current_modes = None):
        self.current_modes = int(self.current_modes_tensor.data.item())
        self.iter = int(self.iter_tensor.data.item())
        if if_starting == True:
            matrix = torch.matmul(self.A.data[:self.current_modes+self.buffer_modes,:].T, self.B.data[:,:self.current_modes+self.buffer_modes].T)
            
            if matrix.dtype != torch.float:
                matrix = matrix.float()

            U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)
            s_sqrt = torch.sqrt(s[:self.current_modes])
            del self._parameters['A']
            del self._parameters['B']
            self.A = nn.Parameter((torch.matmul(U[:,:self.current_modes], torch.diag(s_sqrt)).T).float())
            self.B = nn.Parameter((torch.matmul(torch.diag(s_sqrt), Vh[:self.current_modes,:]).T).float())

            if self.bias_flag:
                bias = self.bias.data
                del self._parameters['bias']
                self.bias = nn.Parameter(bias)
            
            del self._parameters['current_modes_tensor']
            del self._parameters['iter_tensor']
            self.current_modes_tensor = nn.Parameter(torch.tensor([self.current_modes], dtype = torch.float32))
            self.iter_tensor = nn.Parameter(torch.tensor([self.iter], dtype = torch.float32))
            self.current_modes_tensor.requires_grad = False
            self.iter_tensor.requires_grad = False

            return self.current_modes
        
        if if_starting == False:
            size_A = self.A[:current_modes,:].size()
            size_B = self.B[:,:current_modes].size()
            del self._parameters['A']
            del self._parameters['B']
            self.A = nn.Parameter(torch.empty(size_A))
            self.B = nn.Parameter(torch.empty(size_B))

            if self.bias_flag:
                bias = self.bias.data
                del self._parameters['bias']
                self.bias = nn.Parameter(bias)
            
            del self._parameters['current_modes_tensor']
            del self._parameters['iter_tensor']
            self.current_modes_tensor = nn.Parameter(torch.tensor([self.current_modes], dtype = torch.float32))
            self.iter_tensor = nn.Parameter(torch.tensor([self.iter], dtype = torch.float32))
            self.current_modes_tensor.requires_grad = False
            self.iter_tensor.requires_grad = False