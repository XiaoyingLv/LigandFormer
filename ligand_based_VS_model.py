# coding:utf-8
"""
Production model definitions for compound virtual screening based on ligand properties
Created  :   6, 11, 2019
Revised  :  11, 28, 2019  merge attention block modules by Jinjiang
Author   :   Pascal Guo (jinjiang.guo@ghddi.org)
All rights reserved
-------------------------------------------------------------------------------
"""

__author__ = 'd jinjiang.guo'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from pytorch_ext.module import BatchNorm1d, BatchNorm
from torch.nn.utils import weight_norm
import pytorch_ext.functional as EF
import numpy as np, warnings
from graph_ops import GraphConv, GraphReadout
from ligand_based_VS_data_preprocessing import get_atom_feature_dims

from torch.autograd import Variable

# experimental & obsolete models
from ligand_based_VS_model_obsolete import model_2, model_3, model_3v1, model_4, model_4v1, model_4v2, model_4v3
from ligand_based_VS_model_experimental import model_4v5, model_4v6

try:
    import torch_scatter

    torch_scatter_available = True
except ImportError:
    torch_scatter_available = False
    warnings.warn('torch-scatter package is not available, please use `neighbor_op` instead ')

__all__ = [
    'model_0',
    'model_1',
    'model_4v4',
    'model_4v7',
    'Model_Agent',
]

# --- Model Agent ---#
import config as model_config
from pytorch_ext.util import gpickle, get_file_md5
import data_loader
from functools import partial

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss

##---- Spatial Attention Conv --- Jinjing Guo---###
class Spatial_Attention_conv(nn.Module):
    """
    Spatial Attention: giving weights for the importence fo different nodes
    Reffered from paper: Self-Attention Generative Adversarial Networks
    """
    def __init__(self, input_dim, output_dim, kernel_size = 1, padding=0, dilation=1):
        super().__init__()
        
        self.input_dim =    input_dim
        self.output_dim =   output_dim
        self.query = nn.Sequential(
            # nn.LayerNorm(normalized_shape=self.input_dim),
            nn.GroupNorm(self.input_dim, self.input_dim, affine=True),
            nn.Conv1d(self.input_dim, 128, kernel_size =1, padding=0, dilation=1),
            nn.LeakyReLU(0.1, inplace=True)
            ) 
        self.key = nn.Sequential(
            # nn.LayerNorm(normalized_shape=self.input_dim),
            nn.GroupNorm(self.input_dim, self.input_dim, affine=True),
            nn.Conv1d(self.input_dim, 128, kernel_size =1, padding=0, dilation=1),
            nn.LeakyReLU(0.1, inplace=True)
            ) 
        self.val = nn.Sequential(
            # nn.LayerNorm(normalized_shape=self.input_dim),
            nn.GroupNorm(self.input_dim, self.input_dim, affine=True),
            nn.Conv1d(self.input_dim, self.output_dim, kernel_size =1, padding=0, dilation=1),
            nn.LeakyReLU(0.1, inplace=True)
            ) 
        self.apply_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        key_v = self.key(x).squeeze(0).T
        query_v = self.query(x).squeeze(0)
        val_v = self.val(x).squeeze(0).T
        
        spatial_atten = torch.matmul(key_v, query_v)
        spatial_atten = self.apply_softmax(spatial_atten)
        reweight_v = torch.matmul(spatial_atten, val_v)

        return reweight_v, spatial_atten
##---- Spatial Attention  Block for GIN--- Jinjing Guo---###
class Spatial_Attention_Block_GIN(nn.Module):
    """
    Spatial Attention: giving weights for the importence fo different nodes
    Referred from paper: Self-Attention Generative Adversarial Networks
    """
    def __init__(self, input_dim, output_dim, kernel_size = 3, padding=1, dilation=1, viz_att=False):
        super().__init__()
        self.input_dim =    input_dim
        self.output_dim =   output_dim
        self.viz_att =     viz_att

        self.bottle_neck = nn.Sequential(
            # nn.LayerNorm(normalized_shape=self.input_dim),
            nn.GroupNorm(self.input_dim, self.input_dim, affine=True),
            nn.Conv1d(self.input_dim, self.output_dim, kernel_size =1, padding=0, dilation=1),
            nn.LeakyReLU(0.1, inplace=True)
            )     

        self.layer_norm = nn.LayerNorm(normalized_shape=self.output_dim)
        self.leak_relu = nn.LeakyReLU(0.1, inplace=True)
        self.spatial_attention = Spatial_Attention_conv(input_dim=self.output_dim, output_dim=self.output_dim)

    def forward(self, x, dropout=0):
        """
        :param x: node feature matrix, (node_num, feature_dim)
        :return:
        """
        x =x.T # BLC -> BCL
        x = x.unsqueeze_(0)
        x = self.bottle_neck(x) # used for dim deduction
        reweight_x, att_weights =self.spatial_attention(x)
        x = reweight_x.T.unsqueeze_(0)
        x = self.leak_relu(x)
        x = x.squeeze(0) 
        x = x.T
        x = self.layer_norm(x)
        
        if self.viz_att:
            return x, att_weights
        return x

class Model_Agent(object):
    """
    Model agent for handling different model specific events: IO, forward, loss, predict, etc.
    """

    def __init__(self,
                 device=-1,  # int or instance of torch.device
                 model_ver='4v4',  # for the debut version, only 4v4 support is on plan
                 output_dim=2,
                 task='classification',  # {'classification', 'regression'}
                 config=model_config.model_4v4_config(),
                 model_file=None,
                 label_file=None,
                 atom_dict_file=None,
                 load_weights_strict=True,
                 load_all_state_dicts=False,  # if there're multiple state dicts in `model_file`, load them sequentially
                 cipherkey=None,
                 viz_att=True,
                 **kwargs):  # reserved, not used for now
        super().__init__()
        self.model_ver = model_ver
        self.output_dim = output_dim
        self.model_file = model_file
        self.label_file = label_file
        self.atom_dict_file = atom_dict_file
        self.model_file_md5 = None
        self.label_file_md5 = None
        self.atom_dict_file_md5 = None
        self.task = task
        self.config = config
        self.cipherkey = cipherkey
        self.atom_dict = None
        self.label_dict = None
        self.model = None
        self.batch_data_loader = None
        self.load_weights_strict = load_weights_strict
        self.load_all_state_dicts = load_all_state_dicts
        if self.load_all_state_dicts is True:
            self.load_weights_strict = False
        if isinstance(device, torch.device):
            self.device = device
        elif device < 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:%d' % device)
        assert self.task in {'classification', 'regression'}
        self.viz_att = viz_att
        self.initialize()

    @staticmethod
    def load_weight(model_file, cipherkey=None, device=torch.device('cpu')):
        if cipherkey is None or len(cipherkey) == 0:
            model_data = torch.load(model_file, map_location=device)
        else:
            raise NotImplemented('model cipher is not implemented')
        return model_data

    def dump_weight(self, filename, cipherkey=None):
        model_data_to_save = dict()
        model_data_to_save['state_dict'] = self.model.state_dict()
        model_data_to_save['config'] = self.config.__dict__
        if cipherkey is None or len(cipherkey) == 0:
            torch.save(model_data_to_save, filename)
        else:
            raise NotImplemented('model cipher is not implemented')

    def initialize(self):
        if self.atom_dict_file is not None:
            self.atom_dict_file_md5 = get_file_md5(self.atom_dict_file)
            self.atom_dict, _, _ = gpickle.load(self.atom_dict_file)
            if '<unk>' not in self.atom_dict:
                self.atom_dict['<unk>'] = len(self.atom_dict)

        batch_data_loader = getattr(data_loader, 'load_batch_data_%s' % self.model_ver)
        
        if self.model_ver == '4v3':
            return_format = 'neighbor_op' if self.config.use_neighbor_op else 'scatter'
            self.batch_data_loader = partial(batch_data_loader, atom_dict=self.atom_dict,
                                             return_format=return_format)
            self.model = model_4v3(num_embedding=len(self.atom_dict),
                                   output_dim=self.output_dim,
                                   **self.config.__dict__)
        elif self.model_ver == '4v3_1':
            return_format = 'neighbor_op' if self.config.use_neighbor_op else 'scatter'
            self.batch_data_loader = partial(batch_data_loader, return_format=return_format)
            self.model = model_4v3(num_embedding=0,
                                   output_dim=self.output_dim,
                                   **self.config.__dict__)
        elif self.model_ver == '4v4':
            self.batch_data_loader = partial(batch_data_loader, atom_dict=self.atom_dict,
                                             degree_wise=self.config.degree_wise)
            self.model = model_4v4(num_embedding=len(self.atom_dict),
                                   output_dim=self.output_dim, # class num
                                   viz_att=self.viz_att,  
                                   **self.config.__dict__)
                                    
        elif self.model_ver == '4v4_multifeats':
            feature_dims = get_atom_feature_dims()
            self.batch_data_loader = partial(batch_data_loader, atom_dict=self.atom_dict,
                                             degree_wise=self.config.degree_wise)
            self.model = model_4v4_mulitfeats(feature_dims=feature_dims,
                                   output_dim=self.output_dim, # class num
                                   viz_att=self.viz_att,  
                                   **self.config.__dict__)
        elif self.model_ver == '4v4_1':
            self.batch_data_loader = partial(batch_data_loader, degree_wise=self.config.degree_wise)
            self.model = model_4v4(num_embedding=0,
                                   output_dim=self.output_dim,  # class num
                                   **self.config.__dict__)

        elif self.model_ver == '4v5':
            self.batch_data_loader = partial(batch_data_loader, atom_dict=self.atom_dict,
                                             degree_wise=self.config.degree_wise)
            self.model = model_4v5(num_embedding=len(self.atom_dict),
                                   output_dim=self.output_dim,  # class num
                                   **self.config.__dict__)
        elif self.model_ver == '4v5_1':
            self.batch_data_loader = partial(batch_data_loader, degree_wise=self.config.degree_wise)
            self.model = model_4v5(num_embedding=0,
                                   output_dim=self.output_dim,  # class num
                                   **self.config.__dict__)
        elif self.model_ver == '4v6':
            self.batch_data_loader = partial(batch_data_loader, atom_dict=self.atom_dict, degree_wise=False)
            self.model = model_4v6(num_embedding=len(self.atom_dict),
                                   output_dim=self.output_dim,  # class num
                                   **self.config.__dict__)
        elif self.model_ver == '4v6_1':
            self.batch_data_loader = partial(batch_data_loader, degree_wise=False)
            self.model = model_4v6(num_embedding=0,
                                   output_dim=self.output_dim,  # class num
                                   **self.config.__dict__)
        elif self.model_ver == '4v7':
            self.batch_data_loader = partial(batch_data_loader, atom_dict=self.atom_dict,
                                             degree_wise=self.config.degree_wise)
            self.model = model_4v7(num_embedding=len(self.atom_dict),
                                   output_dim=self.output_dim,  # class num
                                   **self.config.__dict__)
        elif self.model_ver == '4v7_1':
            self.batch_data_loader = partial(batch_data_loader, degree_wise=self.config.degree_wise)
            self.model = model_4v7(num_embedding=0,
                                   output_dim=self.output_dim,  # class num
                                   **self.config.__dict__)

        else:
            raise ValueError('self.model_ver = %s not supported' % self.model_ver)

        self.model = self.model.to(self.device)

        if self.model_file is not None:
            self.model_file_md5 = get_file_md5(self.model_file)
            model_data = self.load_weight(self.model_file, self.cipherkey)
            # --- for back compatibility ---#
            if 'state_dict' in model_data:
                state_dict = model_data['state_dict']
                assert model_data[
                           'config'] == self.config.__dict__, "Model config not match, config loaded from model file = %s" % \
                                                              model_data['config']
            else:
                state_dict = model_data
            if isinstance(state_dict, list):
                if not self.load_all_state_dicts:
                    state_dict = [state_dict[0]]
            else:
                state_dict = [state_dict]
            for sdict in state_dict:
                # if self.model_ver == '4v4' and sdict['emb0.weight'].shape[0] > 110:
                #     sdict['emb0.weight'] = sdict['emb0.weight'][0:110, :]
                #     print('embedding weigth trimmed')
                missing_keys, unexpected_keys = self.model.load_state_dict(sdict, strict=self.load_weights_strict)
                if len(missing_keys) > 0:
                    warnings.warn('missing weights = {%s}' % ', '.join(missing_keys))
                if len(unexpected_keys) > 0:
                    warnings.warn('unexpected weights = {%s} ' % ', '.join(unexpected_keys))
        if self.label_file is not None:
            self.label_file_md5 = get_file_md5(self.label_file)
            self.label_dict = gpickle.load(self.label_file)

    def forward(self, batch_data, calc_loss=False, **kwargs):
        graphs = None
        if 'dropout' in kwargs:
            dropout = kwargs['dropout']
        else:
            dropout = 0
        
        if self.model_ver in {'4v3', '4v3_1'}:
            X, padded_neighbors, padded_membership, *Y = batch_data
            X_tensor = torch.from_numpy(X).to(self.device)
            padded_neighbors = torch.from_numpy(padded_neighbors).to(self.device)
            padded_membership = torch.from_numpy(padded_membership).to(self.device)
            scorematrix = self.model.forward(X_tensor, padded_neighbors, padded_membership, dropout)
        elif self.model_ver in {'4v4', '4v4_multifeats', '4v5', '4v4_1', '4v5_1', '4v6', '4v6_1', '4v7', '4v7_1'}:
            if self.config.degree_wise:
                X, edges, membership, degree_slices, *Y = batch_data
            else:
                X, edges, membership, *Y = batch_data
                degree_slices = None
            # batch_data, _, _, _ = gpickle.load(r"C:\Users\lengdawei\Work\Project\graph_pretraining\CLdebug.gpkl")
            X_tensor = torch.from_numpy(X).to(self.device)
            edges = torch.from_numpy(edges).to(self.device)
            membership = torch.from_numpy(membership).to(self.device)
            # for _ in range(5):
            scorematrix, graphs = self.model.forward(X_tensor, edges, membership, dropout, degree_slices=degree_slices)
        else:
            raise ValueError('model_ver = %s not supported' % self.model_ver)
        if calc_loss:
            Y = Y[0]
            Y_tensor = torch.from_numpy(Y).to(self.device)
            if 'class_weight' in kwargs:
                class_weight = kwargs['class_weight']
                class_weight = torch.from_numpy(class_weight).to(self.device)
            else:
                class_weight = None
            if 'sample_weight' in kwargs:
                sample_weight = kwargs['sample_weight']
                sample_weight = torch.from_numpy(sample_weight).to(self.device)
            else:
                sample_weight = None
            if self.task == 'classification':
                if sample_weight is None:
                    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
                    loss = criterion(scorematrix, Y_tensor)
                else:
                    # criterion = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='none')
                    criterion = FocalLoss(gamma=2,size_average=False)
                    loss = criterion(scorematrix, Y_tensor)
                    loss = loss * sample_weight
                    loss = loss.mean()
            else:
                if sample_weight is None:
                    criterion = torch.nn.MSELoss()
                    loss = criterion(scorematrix.squeeze(1), Y_tensor)
                else:
                    criterion = torch.nn.MSELoss(reduction='none')
                    loss = criterion(scorematrix.squeeze(1), Y_tensor)
                    loss = loss * sample_weight
                    loss = loss.mean()
            # gpickle.dump((graphs.detach().cpu().numpy(), scorematrix.detach().cpu().numpy(), loss.detach().cpu().numpy()),'CL_lbvs_debug.gpkl')
            # raise InterruptedError('got CL lbvs')

            return scorematrix, loss
        return scorematrix, graphs

    def predict(self, batch_data, **kwargs):
        self.model.eval()
        scorematrix, *_ = self.forward(batch_data, **kwargs)
        if self.task == 'classification':
            scorematrix = F.softmax(scorematrix, dim=1)
            best_ps, best_idxs = torch.max(scorematrix, dim=1)
            predicted_idxs = best_idxs.detach().cpu().numpy()
            predicted_scores = best_ps.detach().cpu().numpy()
            if self.label_dict is not None:
                predicted_labels = [self.label_dict[idx] for idx in predicted_idxs]
            else:
                predicted_labels = [None] * len(predicted_idxs)
            return predicted_scores, predicted_idxs, predicted_labels
        else:  # regression
            predicted_scores = scorematrix.squeeze(1).cpu().detach().numpy()
            return predicted_scores


# ---- DeepChem's reference model
def degree_wise_neighbor_sum(node_features, deg_adj_list):
    """
    :param node_features: (node_num, feature_dim)
    :param deg_adj_list: list of matrix with shape (node_num_d, degree_num_d) in which `d` means a certain degree
                         each row of the matrix represents the neighbor node indices (excluding the center node)
    :return:
    """
    list_size = len(deg_adj_list)
    neighbor_summed_degreewise = list_size * [None]
    for deg in range(list_size):  # [DV] todo: shoudn't we start from `min_degree` here?
        neighbor_node_features = node_features[
            deg_adj_list[deg]]  # [DV] shape = (n_node_with_given_degree, degree, feature_dim)
        summed_atoms = torch.sum(neighbor_node_features, dim=1,
                                 keepdim=False)  # [DV] shape = (n_node_with_given_degree, feature_dim)
        neighbor_summed_degreewise[deg] = summed_atoms
    return neighbor_summed_degreewise  # len = self.max_degree


def degree_wise_neighbor_max(node_features, deg_slice, deg_adj_list, min_degree=0, max_degree=10):
    """
    refactored from `tensorflow_version.ligandbasedpackage.model_ops.GraphPool`
    :param node_features: 2D tensor with shape (node_num, feature_dim)
    :param deg_slice:     2D tensor with shape (max_deg+1-min_deg,2, 2)
    :param deg_adj_list: list of 2D tensor with shape (node_num, degree_num), len = max_deg+1-min_deg
    :param min_degree: int, 0 or 1
    :param max_degree: int
    :return:
    """
    # maxed_node_feature_list = (max_degree + 1 - min_degree) * [None]
    maxed_node_feature_list = []

    degree_dim = deg_slice.shape[0]  # max_degree + 1, [0, max_degree]
    # for deg in range(1, max_degree+1):
    for deg in range(min_degree, min(max_degree + 1, degree_dim)):
        if deg == 0:  # no neighbors
            self_node_feature = node_features[deg_slice[0, 0]:deg_slice[0, 0] + deg_slice[0, 1],
                                :]  # shape = (n_node_with_given_degree, feature_dim)
            maxed_node_feature_list.append(self_node_feature)
        else:
            if deg_slice[deg, 1] > 0:  # [:,0] for starting index, [:,1] for span size
                start_idx = deg_slice[deg, 0]
                end_idx = deg_slice[deg, 0] + deg_slice[deg, 1]
                self_node_feature = node_features[start_idx:end_idx,
                                    :]  # shape = (n_node_with_given_degree, feature_dim)
                self_node_feature = self_node_feature[:, None, :]  # shape = (n_node_with_given_degree, 1, feature_dim)
                neighbor_node_features = node_features[deg_adj_list[deg - 1],
                                         :]  # shape = (n_node_with_given_degree, degree, feature_dim)
                tmp_node_features = torch.cat([self_node_feature, neighbor_node_features],
                                              dim=1)  # shape = (n_node_with_given_degree, degree+1, feature_dim)
                maxed_node_feature, _ = torch.max(tmp_node_features, dim=1,
                                                  keepdim=False)  # shape = (n_node_with_given_degree, feature_dim)
                # maxed_node_feature_list[deg - min_degree] = maxed_node_feature
                maxed_node_feature_list.append(maxed_node_feature)

    result = torch.cat(maxed_node_feature_list, dim=0)  # todo: pytorch does not handle None properly.
    return result


def graph_readout_DeepChem(node_features, membership, n_graph=None):
    """
    Concatenate mean & max for graph readout
    Implementation difference: 1) no activation affiliated
    :param node_features: 2D tensor with shape (node_num, feature_dim)
    :param membership:    1D tensor with shape (node_num,)
    :param n_graph: int, how many graphs involved in `node_features`
    :return: graph_features, 2D tensor with shape (n_graph, 2 * feature_dim)
    """
    if n_graph is None:
        n_graph = int(max(membership)) + 1
    node_features_for_each_graph = []
    for i in range(n_graph):
        mask = membership == i
        node_features_for_each_graph.append(node_features[mask, :])

    mean_feature_for_each_graph = [
        torch.mean(item, dim=0, keepdim=True)
        for item in node_features_for_each_graph
    ]

    max_feature_for_each_graph = [
        torch.max(item, dim=0, keepdim=True)[0]
        for item in node_features_for_each_graph
    ]
    mean_features_graph = torch.cat(mean_feature_for_each_graph, dim=0)
    max_features_graph = torch.cat(max_feature_for_each_graph, dim=0)
    graph_features = torch.cat([mean_features_graph, max_features_graph], dim=1)  # (n_graph, 2 * feature_dim)

    return graph_features


class Graph_Conv_DeepChem(nn.Module):
    """
    Graph convolution module refactored from `tensorflow_version.ligandbasedpackage.model_ops.GraphConv`
    Implementation difference: 1) no activation affiliated
    neighbor_feature = sum aggregation
    update = affine_1(center_feature) + affine_2(neighbor_feature)
    """

    def __init__(self,
                 in_channels,
                 output_dim,
                 min_deg=0,
                 max_deg=10):
        super().__init__()
        self.in_channels = in_channels  # input feature dimension
        self.output_dim = output_dim  # output feature dimension
        self.min_degree = min_deg
        self.max_degree = max_deg
        self.param_tuple_size = 2 * max_deg + (1 - min_deg)

        self.W_list = nn.ParameterList(
            [Parameter(torch.empty(in_channels, output_dim)) for _ in range(self.param_tuple_size)])
        self.b_list = nn.ParameterList([Parameter(torch.empty(output_dim)) for _ in range(self.param_tuple_size)])
        self.reset_parameters()

    def reset_parameters(self, W_init=nn.init.xavier_uniform_, b_init=nn.init.zeros_):
        for W in self.W_list:
            W_init(W)
        for b in self.b_list:
            b_init(b)

    def forward(self, node_features, deg_slice, deg_adj_list):
        """

        :param node_features: (node_num, feature_dim),
        :param deg_slice: (max_deg+1-min_deg,2, 2),
        :param deg_adj_list: list of tensor with shape=(node_num, degree_num), len = max_deg+1-min_deg
        :return:
        """
        W, b = iter(self.W_list), iter(self.b_list)

        # [DV] aggregate neighbors at each degree level, returned a list of len=self.max_degree
        neighbor_summed_degreewise = degree_wise_neighbor_sum(node_features, deg_adj_list)

        degree_dim = deg_slice.shape[0]  # max_degree + 1, [0, max_degree]
        new_node_feature_collection = []
        # for deg in range(min(self.max_degree + 1, degree_dim)):
        for deg in range(degree_dim):
            if deg == 0:  # no neighbors
                self_node_feature = node_features[deg_slice[0, 0]:deg_slice[0, 0] + deg_slice[0, 1],
                                    :]  # shape = (n_node_with_given_degree, feature_dim)
                out = torch.matmul(self_node_feature, next(W)) + next(b)
                new_node_feature_collection.append(out)
            else:
                # for deg in range(1, min(self.max_degree + 1, degree_dim)):      # [DV] todo: shouldn't we start from `min_degree` here?
                neighbour_feature = neighbor_summed_degreewise[
                    deg - 1]  # [DV] shape = (n_node_with_given_degree, feature_dim)
                self_node_feature = node_features[deg_slice[deg, 0]:deg_slice[deg, 0] + deg_slice[deg, 1],
                                    :]  # [DV] shape = (n_node_with_given_degree, feature_dim)
                # Apply hidden affine to relevant atoms and append
                # [DV] todo: 1) using different affine transforms for `self_atoms` and `rel_atoms` does NOT make any sense
                # [DV] todo: 2) using different affine transforms for different degree graph nodes does NOT make any sense
                # [DV] todo: 3) the only sensible way is to `concat`, not `+` for `self_atoms` and `rel_atoms`
                neighbour_feature = torch.matmul(neighbour_feature, next(W)) + next(b)
                self_node_feature = torch.matmul(self_node_feature, next(W)) + next(b)
                out = neighbour_feature + self_node_feature
                new_node_feature_collection.append(out)

        # Combine all atoms back into the list
        node_features = torch.cat(new_node_feature_collection, dim=0)

        return node_features


class model_0(nn.Module):
    """
    Same model with `tensorflow_version.ligandbasedpackage.graph_models.GraphConvTensorGraph`
    """

    def __init__(self,
                 num_embedding=0,
                 feature_dim=75,
                 graph_conv_layer_size=(64, 64),
                 dense_layer_size=128,
                 dropout=0.0,
                 output_dim=2,
                 min_degree=0,
                 max_degree=10,
                 **kwargs):
        """

        :param graph_conv_layers:
        :param dense_layer_size:
        :param dropout:
        :param output_dim:
        """
        super().__init__()
        self.num_embedding = num_embedding
        if num_embedding > 0:
            self.emb0 = nn.Embedding(num_embeddings=num_embedding, embedding_dim=feature_dim)
        self.graph_conv_layer_size = graph_conv_layer_size
        self.dense_layer_size = dense_layer_size
        self.dropout = dropout
        self.output_dim = output_dim
        self.min_degree = min_degree
        self.max_degree = max_degree
        self.error_bars = True if 'error_bars' in kwargs and kwargs['error_bars'] else False
        self.gconv0 = Graph_Conv_DeepChem(in_channels=feature_dim, output_dim=graph_conv_layer_size[0],
                                          min_deg=self.min_degree, max_deg=self.max_degree)
        self.gconv1 = Graph_Conv_DeepChem(in_channels=graph_conv_layer_size[0], output_dim=graph_conv_layer_size[1],
                                          min_deg=self.min_degree, max_deg=self.max_degree)
        self.bn0 = nn.BatchNorm1d(num_features=graph_conv_layer_size[0])
        self.bn1 = nn.BatchNorm1d(num_features=graph_conv_layer_size[1])
        self.bn2 = nn.BatchNorm1d(num_features=dense_layer_size)
        self.dense0 = nn.Linear(in_features=graph_conv_layer_size[1], out_features=dense_layer_size)
        self.dense1 = nn.Linear(in_features=2 * dense_layer_size, out_features=output_dim)

    def forward(self, x, degree_slice, membership, deg_adj_list):
        """
        Forward pass
        :param x: either feature matrix of float tensor(node_num, feature_dim), or node index tensor (node_num,)
        :param degree_slice: int tensor matrix, (max_deg+1-min_deg, 2)
        :param membership: int tensor, (node_num,)
        :param deg_adj_list: list of int tensor with shape=(node_num, degree_num), len = max_deg+1-min_deg
        :return: un-normalized class distribution
        """
        if self.num_embedding > 0:
            x = self.emb0(x)  # (node_num, ) -> (node_num, feature_dim)
        x = self.gconv0(x, degree_slice, deg_adj_list)
        x = torch.relu(x)
        x = self.bn0(x)
        x = degree_wise_neighbor_max(x, degree_slice, deg_adj_list, min_degree=self.min_degree,
                                     max_degree=self.max_degree)
        x = self.gconv1(x, degree_slice, deg_adj_list)
        x = torch.relu(x)
        x = self.bn1(x)
        x = degree_wise_neighbor_max(x, degree_slice, deg_adj_list, min_degree=self.min_degree,
                                     max_degree=self.max_degree)
        x = self.dense0(x)
        x = self.bn2(x)
        x = torch.dropout(x, p=self.dropout, train=self.training)
        x = graph_readout_DeepChem(x, membership)
        x = torch.tanh(x)
        x = self.dense1(x)
        # x = torch.log_softmax(x, dim=1)
        return x


# ---- RNN baseline
class model_1(nn.Module):
    """
    A basic RNN model served as baseline
    """

    def __init__(self,
                 num_embedding=0,
                 feature_dim=75,
                 hidden_size=128,
                 bidirectional=True,
                 dropout=0.5,
                 output_dim=2,
                 **kwargs):
        """

        :param num_embedding: if >0, input is assumed to be SMILES string; else input will be assumed to be atom feature sequence
        :param feature_dim:
        :param hidden_size:
        :param num_layers:
        :param bidirectional:
        :param dropout:
        :param output_dim:
        """
        super().__init__()
        self.num_embedding = num_embedding
        num_directions = 2 if bidirectional else 1
        if num_embedding > 0:
            self.emb0 = nn.Embedding(num_embeddings=num_embedding, embedding_dim=feature_dim)
        self.lstm0 = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size, num_layers=1,
                             bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.lstm1 = nn.LSTM(input_size=num_directions * hidden_size, hidden_size=hidden_size, num_layers=1,
                             bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=num_directions * hidden_size, hidden_size=hidden_size, num_layers=1,
                             bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.dense0 = nn.Linear(in_features=num_directions * hidden_size, out_features=output_dim)

    def forward(self, x):
        """
        :param x: (B, T) if self.num_embedding > 0 else (B, T, D)
        :return:
        """
        if self.num_embedding > 0:
            x = self.emb0.forward(x)
        x, _ = self.lstm0.forward(x)  # (B, T, num_direction*hidden_size)
        x = torch.transpose(x, 1, 2)  # (B, T, D)->(B, D, T)
        x = F.max_pool1d(x, kernel_size=2)  # (B, D, T) -> (B, D, T//2)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm1.forward(x)
        x = torch.transpose(x, 1, 2)  # (B, T, D)->(B, D, T)
        x = F.max_pool1d(x, kernel_size=2)  # (B, D, T) -> (B, D, T//2)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm2.forward(x)
        x = x[:, -1, :]  # (B, D)
        x = self.dense0.forward(x)
        return x


# --- model_4 series
class model_4v4(nn.Module):
    """
    HAG-Net: hybrid aggregation graph network
    """

    def __init__(self,
                 num_embedding=0,
                 block_num=5,
                 input_dim=75,
                 hidden_dim=256,
                 output_dim=2,  # class num
                 degree_wise=False,
                 max_degree=1,
                 aggregation_methods=('max', 'sum'),
                 multiple_aggregation_merge_method='sum',
                 affine_before_merge=False,
                 node_feature_update_method='rnn',
                 readout_methods=('rnn-sum-max',),
                 multiple_readout_merge_method='sum',
                 add_dense_connection=True,  # whether add dense connection among the blocks
                 pyramid_feature=True,
                 slim=True,
                 **kwargs
                 ):
        super().__init__()
        self.num_embedding = num_embedding
        if num_embedding > 0:
            self.emb0 = nn.Embedding(num_embeddings=num_embedding, embedding_dim=input_dim)
        self.block_num = block_num
        self.degree_wise = degree_wise
        self.max_degree = max_degree
        self.aggregation_methods = aggregation_methods
        self.multiple_aggregation_merge = multiple_aggregation_merge_method
        self.readout_methods = readout_methods
        self.add_dense_connection = add_dense_connection
        self.pyramid_feature = pyramid_feature
        self.slim = slim
        self.classifier_dim = input_dim
        self.blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.blocks.append(GraphConv(input_dim=input_dim, output_dim=input_dim,
                                         aggregation_methods=aggregation_methods,
                                         multiple_aggregation_merge_method=multiple_aggregation_merge_method,
                                         affine_before_merge=affine_before_merge,
                                         update_method=node_feature_update_method,
                                         degree_wise=degree_wise,
                                         max_degree=max_degree,
                                         backbone='default',
                                         **kwargs,
                                         ))
        self.readout_ops = nn.ModuleList()
        if self.pyramid_feature:
            readout_block_num = self.block_num + 1
        else:
            readout_block_num = 1
        self.readout_block_num = readout_block_num
        for i in range(readout_block_num):
            self.readout_ops.append(GraphReadout(readout_methods=self.readout_methods, input_dim=input_dim,
                                                 multiple_readout_merge_method=multiple_readout_merge_method,
                                                 affine_before_merge=affine_before_merge,
                                                 degree_wise=degree_wise,
                                                 max_degree=max_degree,
                                                 **kwargs))
            if self.slim:
                break

        readout_dim = input_dim * readout_block_num
        if self.degree_wise:
            readout_dim *= (self.max_degree + 1)
        if 'classifier_dim' in kwargs:
            self.classifier_dim = kwargs['classifier_dim']
        self.dense0 = nn.Linear(in_features=readout_dim, out_features=hidden_dim)
        self.dense1 = nn.Linear(in_features=hidden_dim, out_features=self.classifier_dim)
        self.dense2 = nn.Linear(in_features=self.classifier_dim, out_features=output_dim)

        if 'norm_method' in kwargs:
            norm_method = kwargs['norm_method'].lower()
        else:
            norm_method = 'bn'
        if norm_method == 'ln':
            self.bn0 = nn.LayerNorm(normalized_shape=hidden_dim)
            self.bn1 = nn.LayerNorm(normalized_shape=hidden_dim)
        elif norm_method == 'bn_notrack':
            self.bn0 = BatchNorm1d(num_features=hidden_dim, track_running_stats=False)
            self.bn1 = BatchNorm1d(num_features=hidden_dim, track_running_stats=False)
        elif norm_method == 'none':
            self.bn0 = None
            self.bn1 = None
        elif norm_method == 'wn':
            self.dense0 = weight_norm(self.dense0, name='weight', dim=0)
            self.dense1 = weight_norm(self.dense1, name='weight', dim=0)
            self.bn0 = None
            self.bn1 = None
        elif norm_method == 'newbn':
            self.bn0 = BatchNorm(input_shape=(None, hidden_dim))
            self.bn1 = BatchNorm(input_shape=(None, hidden_dim))
        else:  # default = bn
            self.bn0 = BatchNorm1d(num_features=hidden_dim, momentum=0.01)
            self.bn1 = BatchNorm1d(num_features=hidden_dim, momentum=0.01)

    def forward(self, x, edges=None, membership=None, dropout=0.0, degree_slices=None):
        """
        :param x: (node_num,) int64 if embedding is enabled; (node_num, feature_dim) else
        :param edges: (2, edge_num), int64, each column in format of (neighbor_node, center_node)
        :param membership: (node_num,), int64, representing to which graph the i_th node belongs
        :param dropout: dropout value
        :param degree_slices: (max_degree_in_batch, 2), each row in format of (start_idx, end_idx), in which '*_idx' corresponds
                              to edges indices; i.e., each row is the span of edges whose center node is of the same degree,
                              required when self.degree_wise = True, otherwise leave it to None
        :return: x, (batch_size, class_num)
        """
        # dropout
        drop_aggregation = dropout
        drop_readout = dropout
        drop_classification = dropout

        if self.num_embedding > 0:
            x = self.emb0(x)
        # --- aggregation ---#
        if self.pyramid_feature:
            hiddens = [x]
        block_input = x
        for i in range(self.block_num):
            x = self.blocks[i](x=block_input, edges=edges,
                               include_self_in_neighbor=False,
                               dropout=drop_aggregation,
                               degree_slices=degree_slices)
            if self.add_dense_connection:
                block_input = block_input + x
            else:
                block_input = x
            if self.pyramid_feature:
                hiddens.append(x)
        # --- readout ---#
        if self.pyramid_feature:
            graph_representations = []
            for i in range(self.block_num + 1):
                idx = 0 if self.slim else i
                pooled = self.readout_ops[idx](hiddens[i], membership, degree_slices=degree_slices, dropout=drop_readout)
                graph_representations.append(pooled)
            x = torch.cat(graph_representations, dim=1)
        else:

            x = self.readout_ops[0](x, membership, degree_slices=degree_slices, dropout=drop_readout)

        # --- classification ---#
        graphs = x
        x = self.dense0(x)
        x = self.bn0(x)
        x = F.gelu(x)
        if self.training and drop_classification > 0.0:
            x = EF.dropout(x, p=drop_classification * 2, rescale=True)
        x = self.dense1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        if self.training and drop_classification > 0.0:
            x = EF.dropout(x, p=drop_classification * 2, rescale=True)
        x = self.dense2(x)
        return x, graphs
        
######--------4v4 for multi features -------####
class model_4v4_mulitfeats(nn.Module):
    """
    HAG-Net: hybrid aggregation graph network with multiple features and self-attention
    """

    def __init__(self,
                 feature_dims = [0],
                 block_num=5,
                 input_dim=75,
                 hidden_dim=256,
                 output_dim=2,  # class num
                 degree_wise=False,
                 max_degree=1,
                 aggregation_methods=('max', 'sum'),
                 multiple_aggregation_merge_method='sum',
                 affine_before_merge=False,
                 node_feature_update_method='rnn',
                 readout_methods=('rnn-sum-max',),
                 multiple_readout_merge_method='sum',
                 add_dense_connection=True,  # whether add dense connection among the blocks
                 pyramid_feature=True,
                 slim=True,
                 viz_att=False,
                 **kwargs
                 ):
        super().__init__()
        self.feature_dims = feature_dims
        self.block_num = block_num
        self.degree_wise = degree_wise
        self.max_degree = max_degree
        self.aggregation_methods = aggregation_methods
        self.multiple_aggregation_merge = multiple_aggregation_merge_method
        self.readout_methods = readout_methods
        self.add_dense_connection = add_dense_connection
        self.pyramid_feature = pyramid_feature
        self.slim = slim
        self.viz_att = viz_att
        self.classifier_dim = input_dim
        self.blocks = nn.ModuleList()
        self.embeds = nn.ModuleList()
        self.spatial_atts = nn.ModuleList()
        for i, dim in enumerate(self.feature_dims):
            emb = torch.nn.Embedding(num_embeddings=dim, embedding_dim=input_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.embeds.append(emb)
        for i in range(self.block_num):
            self.blocks.append(GraphConv(input_dim=input_dim, output_dim=input_dim,
                                         aggregation_methods=aggregation_methods,
                                         multiple_aggregation_merge_method=multiple_aggregation_merge_method,
                                         affine_before_merge=affine_before_merge,
                                         update_method=node_feature_update_method,
                                         degree_wise=degree_wise,
                                         max_degree=max_degree,
                                         backbone='default',
                                         **kwargs,
                                         ))
            spatial_att_input_dim = (i + 2) * input_dim
            self.spatial_atts.append(Spatial_Attention_Block_GIN(spatial_att_input_dim, input_dim, viz_att=self.viz_att))
        self.readout_ops = nn.ModuleList()
        self.atten_feat_readout = GraphReadout(readout_methods=self.readout_methods, input_dim=input_dim,
                                                 multiple_readout_merge_method=multiple_readout_merge_method,
                                                 affine_before_merge=affine_before_merge,
                                                 degree_wise=degree_wise,
                                                 max_degree=max_degree,
                                                 **kwargs)
        if self.pyramid_feature:
            readout_block_num = self.block_num + 1
        else:
            readout_block_num = 1
        self.readout_block_num = readout_block_num
        for i in range(readout_block_num):
            self.readout_ops.append(GraphReadout(readout_methods=self.readout_methods, input_dim=input_dim,
                                                 multiple_readout_merge_method=multiple_readout_merge_method,
                                                 affine_before_merge=affine_before_merge,
                                                 degree_wise=degree_wise,
                                                 max_degree=max_degree,
                                                 **kwargs))
            if self.slim:
                break

        readout_dim = input_dim * readout_block_num
        if self.degree_wise:
            readout_dim *= (self.max_degree + 1)
        if 'classifier_dim' in kwargs:
            self.classifier_dim = kwargs['classifier_dim']
        self.dense0 = nn.Linear(in_features=readout_dim, out_features=hidden_dim)
        self.dense1 = nn.Linear(in_features=hidden_dim, out_features=self.classifier_dim)
        self.dense2 = nn.Linear(in_features=self.classifier_dim, out_features=output_dim)

        if 'norm_method' in kwargs:
            norm_method = kwargs['norm_method'].lower()
        else:
            norm_method = 'bn'
        if norm_method == 'ln':
            self.bn0 = nn.LayerNorm(normalized_shape=hidden_dim)
            self.bn1 = nn.LayerNorm(normalized_shape=hidden_dim)
        elif norm_method == 'bn_notrack':
            self.bn0 = BatchNorm1d(num_features=hidden_dim, track_running_stats=False)
            self.bn1 = BatchNorm1d(num_features=hidden_dim, track_running_stats=False)
        elif norm_method == 'none':
            self.bn0 = None
            self.bn1 = None
        elif norm_method == 'wn':
            self.dense0 = weight_norm(self.dense0, name='weight', dim=0)
            self.dense1 = weight_norm(self.dense1, name='weight', dim=0)
            self.bn0 = None
            self.bn1 = None
        elif norm_method == 'newbn':
            self.bn0 = BatchNorm(input_shape=(None, hidden_dim))
            self.bn1 = BatchNorm(input_shape=(None, hidden_dim))
        else:  # default = bn
            self.bn0 = BatchNorm1d(num_features=hidden_dim, momentum=0.01)
            self.bn1 = BatchNorm1d(num_features=hidden_dim, momentum=0.01)

    def forward(self, x, edges=None, membership=None, dropout=0.0, degree_slices=None):
        """
        :param x: (node_num,) int64 if embedding is enabled; (node_num, feature_dim) else
        :param edges: (2, edge_num), int64, each column in format of (neighbor_node, center_node)
        :param membership: (node_num,), int64, representing to which graph the i_th node belongs
        :param dropout: dropout value
        :param degree_slices: (max_degree_in_batch, 2), each row in format of (start_idx, end_idx), in which '*_idx' corresponds
                              to edges indices; i.e., each row is the span of edges whose center node is of the same degree,
                              required when self.degree_wise = True, otherwise leave it to None
        :return: x, (batch_size, class_num)
        """
        x_embedings = self.embeds[0](x[:,0])
        for i in range(1, x.shape[1]):
            x_embedings += self.embeds[i](x[:,i])
            # x_embedings = torch.cat((x_embedings,self.embeds[i](x[:,i])), dim=1)
        x = x_embedings
        # --- aggregation ---#
        if self.pyramid_feature:
            hiddens = [x]
        # if self.viz_att:
        block_fusion = x
        block_input = x
        att_map = []
        for i in range(self.block_num):
            x = self.blocks[i](x=block_input, edges=edges,
                               include_self_in_neighbor=False,
                               dropout=dropout,
                               degree_slices=degree_slices)
            if self.add_dense_connection:
                block_input = block_input + x
            else:
                block_input = x
            block_fusion = torch.cat((block_fusion,x), dim=1)
            if self.viz_att:
                reweight_block_fusion, att_map_tem = self.spatial_atts[i](block_fusion)
                att_map.append(att_map_tem)
                x = block_input + reweight_block_fusion
            else:
                reweight_block_fusion = self.spatial_atts[i](block_fusion)
                x = block_input + reweight_block_fusion
            
            if self.pyramid_feature:
                hiddens.append(x)          
        # --- readout ---#
        if self.pyramid_feature:
            graph_representations = []
            for i in range(self.block_num + 1):
                idx = 0 if self.slim else i
                pooled = self.readout_ops[idx](hiddens[i], membership, degree_slices=degree_slices, dropout=dropout)
                graph_representations.append(pooled)
            x = torch.cat(graph_representations, dim=1)
        else:

            x = self.readout_ops[0](x, membership, degree_slices=degree_slices, dropout=dropout)

        # --- classification ---#
        if self.viz_att:
            graphs = att_map # attention maps are visualized here
        else:
            graphs = x
        x = self.dense0(x)
        x = self.bn0(x)
        x = F.gelu(x)
        if self.training and dropout > 0.0:
            x = EF.dropout(x, p=dropout * 2, rescale=True)
        x = self.dense1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        if self.training and dropout > 0.0:
            x = EF.dropout(x, p=dropout * 2, rescale=True)
        x = self.dense2(x)
        return x, graphs