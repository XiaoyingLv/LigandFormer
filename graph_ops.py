# coding:utf-8
"""
  Generic graph convolution ops (torch)
  Created   :  11, 26, 2019
  Revised   :   7, 16, 2020  improve `.reset_parameters()` for GraphConv & GraphReadout
  All rights reserved

  * Difference between LayerNorm(D) and BatchNorm1d(D)
           LayerNorm(D) = (x - x.mean(dim=1,keepdim=True))/torch.sqrt(x.var(dim=1, keepdim=True)+1e-5)
           BatchNorm(D) = (x - x.mean(dim=0))/torch.sqrt(x.var(dim=0)+1e-5)
"""
__author__ = 'dawei.leng'

import torch, time
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from pytorch_ext.module import BatchNorm1d, BatchNorm
import pytorch_ext.functional as EF
from torch.nn.utils import weight_norm
import numpy as np, warnings
import torch_scatter

if torch_scatter.__version__.startswith('1.'):
    raise RuntimeError('please upgrade your torch_scatter to version > 2.0')
else:  # for torch_scatter > 2.0
    from torch_scatter import scatter, scatter_softmax


############################################################
#  Utility functions, from pytorch-geometric.utils.loop
#  only used along with torch-scatter package
############################################################
def maybe_num_nodes(index, num_nodes=None):
    # type: (Tensor, Optional[int]) -> int
    return index.max().item() + 1 if num_nodes is None else num_nodes


def contains_self_loops(edge_index):
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` contains
    self-loops.

    Args:
        edge_index (LongTensor): The edge indices.

    :rtype: bool
    """
    row, col = edge_index
    mask = row == col
    return mask.sum().item() > 0


def remove_self_loops(edge_index, edge_attr=None):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    row, col = edge_index
    mask = row != col
    edge_attr = edge_attr if edge_attr is None else edge_attr[mask]
    edge_index = edge_index[:, mask]

    return edge_index, edge_attr


def segregate_self_loops(edge_index, edge_attr=None):
    r"""Segregates self-loops from the graph.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`, :class:`LongTensor`,
        :class:`Tensor`)
    """

    mask = edge_index[0] != edge_index[1]
    inv_mask = ~mask

    loop_edge_index = edge_index[:, inv_mask]
    loop_edge_attr = None if edge_attr is None else edge_attr[inv_mask]
    edge_index = edge_index[:, mask]
    edge_attr = None if edge_attr is None else edge_attr[mask]

    return edge_index, edge_attr, loop_edge_index, loop_edge_attr


def add_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted, self-loops will be added with edge weights
    denoted by :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        fill_value (int, optional): If :obj:`edge_weight` is not :obj:`None`,
            will add self-loops with edge weights of :obj:`fill_value` to the
            graph. (default: :obj:`1`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    loop_index = torch.arange(0, num_nodes, dtype=torch.long,
                              device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((num_nodes,), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight


def add_remaining_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    # type: (Tensor, Optional[Tensor], int, Optional[int]) -> Tuple[Tensor, Optional[Tensor]]
    r"""Adds remaining self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted and already contains a few self-loops, only
    non-existent self-loops will be added with edge weights denoted by
    :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        fill_value (float, optional): If :obj:`edge_weight` is not :obj:`None`,
            will add self-loops with edge weights of :obj:`fill_value` to the
            graph. (default: :obj:`1`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)  # [DV] guess node number if `num_nodes` is not given
    row, col = edge_index[0, :], edge_index[1,
                                 :]  # [DV] edge_index.shape = (2, edge_num), row = source node index, col = target node index

    mask = row != col  # [DV] mask for non-self-connection

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)  # [DV] edge number
        inv_mask = ~mask

        loop_weight = torch.full(
            (num_nodes,), fill_value,
            dtype=None if edge_weight is None else edge_weight.dtype,
            device=edge_index.device)
        remaining_edge_weight = edge_weight[inv_mask]
        if remaining_edge_weight.numel() > 0:
            loop_weight[row[inv_mask]] = remaining_edge_weight
        edge_weight = torch.cat([edge_weight[mask], loop_weight],
                                dim=0)  # [DV] keep original weights meanwhile use `fill_value` for all added self connections

    loop_index = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index],
                           dim=1)  # [DV] all non-self connections + all self connections

    return edge_index, edge_weight


def norm_by_degree(edge_index, num_nodes, edge_weight=None, self_connection_weight=1.0, dtype=None):
    """
    Normalize edge weights by node (out-)degrees
    Only used along with torch-scatter package
    Modified from torch_geometric.nn.conv.gcn_conv.norm()
    :param edge_index: (2, edge_num), each column in format of (neighbor_node, center_node)
    :param num_nodes: number of nodes in graph
    :param edge_weight: (edge_num,). The normalization is done by weighting each edge with normalized weights. If not given,
                        `edge_weight` will initialized as all ones.
    :param self_connection_weight: weight value for self-connections
    :param dtype: dtype of `edge_weight`
    :return: edge_index (with self connections added), normalized_edge_weight
    """
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)  # [DV] all 1s if not given

    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, self_connection_weight, num_nodes)

    row, col = edge_index  # [DV] nodes index for neighbor & center
    deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes,
                  reduce='sum')  # [DV] (edge_num,) calc degree_out for each source node
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    normalized_edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_index, normalized_edge_weight


############################################################
#                  Graph operations                        #
############################################################

def neighbor_op(x, padded_neighbors, padded_edge_weights=None, op='max', include_self=False,
                self_connection_weight=1.0):
    """
    Generic neighborhood operations
    For speed comparison, according to test.test_torch_scatter results:
        1) On CPU, time cost of `neighbor_op` is about 1.9 ~ 4.8 times vs torch.spmm
        2) On GPU, time cost of `neighbor_op` is about 0.2 ~ 0.3 times vs torch.spmm
        3) On CPU, time cost of `neighbor_op` is about 1.5 ~ 3.8 times vs torch-scatter
        4) On GPU, time cost of `neighbor_op` is about 0.9 ~ 1.0 times vs torch-scatter
    To conclude:
        1) if available, torch-scatter is generally preferred over others
        2) else, neighbor_op is generally preferred over torch.spmm
    :param x: node features, (N, D).
    :param padded_neighbors: (N, max_degree_in_batch), int64, each row represents the 1-hop neighbors for node_i; `-1`
                             are padded to indicate invalid values, self nodes are presumed excluded.
    :param padded_edge_weights: (N, max_degree_in_batch), float32, storing edge weights for the Nth center node and its
                                neighbor nodes, '1's are padded at the end of each row. By default, it's just all ones.
    :param include_self: whether append self nodes when perform neighborhood operations. If `padded_neighbors` already
                         include self nodes, remember set this flag to False.
    :param op: supported neighborhood operations include {'max', 'min', 'sum', 'mean'}.
               when 'mean', possible `inf` values will be replaced with 0;
    :param self_connection_weight: edge weight for self connections
    :return:
    """
    if include_self:
        N = x.shape[0]
        self_idxs = torch.arange(0, N, dtype=padded_neighbors.dtype, device=padded_neighbors.device).view(N, 1)
        padded_neighbors = torch.cat([self_idxs, padded_neighbors], dim=1)
        if padded_edge_weights is not None:
            self_connection_weights = torch.ones(size=(N, 1), dtype=padded_edge_weights.dtype,
                                                 device=padded_edge_weights.device) * self_connection_weight
            padded_edge_weights = torch.cat([self_connection_weights, padded_edge_weights], dim=1)

    if op == 'max':
        dummy = torch.min(x, dim=0)[0]  # (D,)
        x_with_dummy = torch.cat([x, dummy.reshape((1, -1))])  # (N+1, D)
        x_neighbors = x_with_dummy[padded_neighbors]  # (N, max_degree_in_batch, D)
        if padded_edge_weights is not None:
            x_neighbors = x_neighbors * padded_edge_weights[:, :, None]
        result = torch.max(x_neighbors, dim=1)[0]  # (N, D)
    elif op == 'min':
        dummy = torch.max(x, dim=0)[0]  # (D,)
        x_with_dummy = torch.cat([x, dummy.reshape((1, -1))])  # (N+1, D)
        x_neighbors = x_with_dummy[padded_neighbors]  # (N, max_degree_in_batch, D)
        if padded_edge_weights is not None:
            x_neighbors = x_neighbors * padded_edge_weights[:, :, None]
        result = torch.min(x_neighbors, dim=1)[0]  # (N, D)
    elif op in {'sum', 'mean'}:
        D = x.shape[1]
        dummy = x.new_zeros(size=(1, D))
        x_with_dummy = torch.cat([x, dummy], dim=0)  # (N+1, D)
        x_neighbors = x_with_dummy[padded_neighbors]  # (N, max_degree_in_batch, D)
        if padded_edge_weights is not None:
            x_neighbors = x_neighbors * padded_edge_weights[:, :, None]
        result = torch.sum(x_neighbors, dim=1)  # (N, D)
        if op == 'mean':
            mask = padded_neighbors >= 0
            nums = mask.sum(dim=1).view(-1, 1)
            result = result / nums
            result[result == float('inf')] = 0.0
    else:
        raise ValueError('Invalid op = %s' % op)
    return result


class GraphConv(nn.Module):
    """
    A generic graph *convolution* module, degree-wise support removed
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 aggregation_methods=('sum', 'max'),  # {'sum', 'mean', 'max', 'min', 'att'}
                 multiple_aggregation_merge_method='cat',  # {'cat', 'sum'}
                 affine_before_merge=False,
                 update_method='cat',  # {'cat', 'sum', 'rnn', 'max'}
                 backbone='default',
                 **kwargs
                 ):
        """
        :param input_dim:
        :param output_dim:
        :param aggregation_methods: tuple of strings in  {'sum', 'mean', 'max', 'min', 'att'}
        :param multiple_aggregation_merge_method: {'cat', 'sum'}, how their results should be merged
                                                  if there are multiple aggregation methods simultaneously
        :param affine_before_merge: if True, output of each neighborhood aggregation method will be further
                                    affine-transformed before they are merged
        :param update_method: {'cat', 'sum', 'rnn', 'max'}, how the center node feature should be merged with aggregated neighbor feature
        :param backbone: nn.Module for feature transformation, a two-layer dense module will be used by default, you can
                         set it to `None` to disable this transformation. Class function `.reset_parameters()` is required to be defined.
        :param kwargs:  1) head_num: attention head number, for `att` aggregation method, default = 1
                        2) att_mode: {'combo', 'single'}, specify attention mode for `att` aggregation method. The `att`
                           method is basically correlating node features with the attention vector, this correlation can
                           be done at single node level or at neighbor-center combination level. For the latter mode, attention
                           is done on concatenation of each tuple of (neighbor, center) node features.
                        3) norm_method: {'newbn', 'bn', 'bn_notrack', 'ln', 'wn', 'none'}, specify what normalization method to use. 'bn' = batch normalization
                           'bn_notrack' = batch normalization without tracking running statistics, 'ln'= layer normalization,
                           'wn' = weight normalization
                        4) hidden_dim: hidden dim for backbone layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.affine_before_merge = affine_before_merge
        self.aggregation_methods = []
        for item in aggregation_methods:
            item = item.lower()
            if item not in {'sum', 'mean', 'max', 'min', 'att'}:
                raise ValueError("aggregation method should be in {'sum', 'mean', 'max', 'min', 'att'}")
            self.aggregation_methods.append(item)
            if item == 'att':
                if 'head_num' in kwargs:
                    self.head_num = kwargs['head_num']
                else:
                    self.head_num = 1
                assert self.input_dim % self.head_num == 0, 'input_dim must be multiple of head_num'
                if 'att_mode' in kwargs:
                    self.att_mode = kwargs['att_mode']
                else:
                    self.att_mode = 'combo'
                assert self.att_mode in {'single', 'combo'}
                if self.att_mode == 'single':
                    self.att_weight = Parameter(torch.empty(size=(1, self.head_num, self.input_dim // self.head_num)))
                else:
                    self.att_weight = Parameter(
                        torch.empty(size=(1, self.head_num, 2 * self.input_dim // self.head_num)))
        self.multiple_aggregation_merge_method = multiple_aggregation_merge_method.lower()
        assert self.multiple_aggregation_merge_method in {'cat', 'sum'}
        aggregation_num = len(self.aggregation_methods)
        if self.affine_before_merge:
            self.affine_transforms = nn.ModuleList()
            for i in range(aggregation_num):
                self.affine_transforms.append(nn.Linear(in_features=input_dim, out_features=input_dim))
        if aggregation_num > 1:
            if self.multiple_aggregation_merge_method == 'sum':
                pass
            else:
                self.merge_layer = nn.Linear(in_features=input_dim * aggregation_num, out_features=input_dim)
        self.update_method = update_method.lower()
        assert self.update_method in {'cat', 'sum', 'rnn', 'max'}
        if self.update_method == 'rnn':
            self.rnn = nn.GRUCell(input_size=input_dim, hidden_size=input_dim)

        if backbone is not None and backbone.lower() == 'default':
            if 'hidden_dim' in kwargs:
                hidden_dim = kwargs['hidden_dim']
            else:
                hidden_dim = 256
            if self.update_method == 'cat':
                backbone_input_dim = 2 * input_dim
            else:
                backbone_input_dim = input_dim
            backbone_dense0 = nn.Linear(in_features=backbone_input_dim, out_features=hidden_dim)
            backbone_dense1 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

            norm_method = 'bn'
            if 'norm_method' in kwargs:
                norm_method = kwargs['norm_method'].lower()
            if norm_method == 'ln':
                norm_layer0 = nn.LayerNorm(normalized_shape=hidden_dim)
                norm_layer1 = nn.LayerNorm(normalized_shape=output_dim)
            elif norm_method == 'bn_notrack':
                norm_layer0 = BatchNorm1d(num_features=hidden_dim, track_running_stats=False)
                norm_layer1 = BatchNorm1d(num_features=output_dim, track_running_stats=False)
            elif norm_method == 'none':
                norm_layer0 = None
                norm_layer1 = None
            elif norm_method == 'wn':
                backbone_dense0 = weight_norm(backbone_dense0, name='weight', dim=0)
                backbone_dense1 = weight_norm(backbone_dense1, name='weight', dim=0)
                norm_layer0 = None
                norm_layer1 = None
            elif norm_method == 'newbn':
                norm_layer0 = BatchNorm(input_shape=(None, hidden_dim))
                norm_layer1 = BatchNorm(input_shape=(None, output_dim))
            else:  # default = bn
                norm_layer0 = BatchNorm1d(num_features=hidden_dim, momentum=0.01)
                norm_layer1 = BatchNorm1d(num_features=output_dim, momentum=0.01)

            if norm_layer0 is None:
                self.backbone = nn.Sequential(backbone_dense0,
                                              nn.LeakyReLU(),
                                              backbone_dense1,
                                              nn.LeakyReLU(),
                                              )
            else:
                self.backbone = nn.Sequential(backbone_dense0,
                                              nn.LeakyReLU(),
                                              norm_layer0,
                                              backbone_dense1,
                                              nn.LeakyReLU(),
                                              norm_layer1,
                                              )
        else:
            self.backbone = backbone

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'att_weight'):
            nn.init.xavier_normal_(self.att_weight)
        if hasattr(self, 'affine_transforms'):
            for module in self.affine_transforms:
                module.reset_parameters()
        if hasattr(self, 'merge_layer'):
            self.merge_layer.reset_parameters()
        if hasattr(self, 'rnn'):
            self.rnn.reset_parameters()
        if isinstance(self.backbone, nn.Sequential):
            for module in self.backbone:
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
        elif self.backbone is not None:
            self.backbone.reset_parameters()

    def forward(self, x, edges, edge_weights=None, include_self_in_neighbor=False, **kwargs):
        """
        :param x: (node_num, D)
        :param edges: (2, edge_num), int64, each column in format of (neighbor_node, center_node).
        :param edge_weights: (edge_num,), edge weights, order the same with `edges`
        :param include_self_in_neighbor: when performing neighborhood operations, whether include self (center) nodes
        :param kwargs: 1) dropout: dropout value in range [0, 1)
        :return:
        """
        node_num, feature_dim = x.shape
        edge_num = edges.shape[1]
        if include_self_in_neighbor:
            edges, edge_weights = add_remaining_self_loops(edges, edge_weights, num_nodes=node_num)
        x_neighbor = x[edges[0, :]]
        if edge_weights is not None:
            x_neighbor = x_neighbor * edge_weights[:, None]
        dropout = 0.0
        if 'dropout' in kwargs:
            dropout = kwargs['dropout']
        # ---- neighborhood aggregation ----#
        aggr_outputs = []
        scatter_index = edges[1, :]
        for i, aggr_method in enumerate(self.aggregation_methods):
            if aggr_method == 'max':
                scatter_input = EF.dropout(x_neighbor, p=dropout, shared_axes=(-1,), rescale=False,
                                           fill_value=0.0) if dropout > 0.0 else x_neighbor
                x_aggregated = scatter(scatter_input, scatter_index, dim=0, dim_size=node_num,
                                       reduce='max')  # (node_num, feature_dim)
            elif aggr_method == 'min':
                scatter_input = EF.dropout(x_neighbor, p=dropout, shared_axes=(-1,), rescale=False,
                                           fill_value=0.0) if dropout > 0.0 else x_neighbor
                x_aggregated = scatter(scatter_input, scatter_index, dim=0, dim_size=node_num, reduce='min')
            elif aggr_method == 'mean':
                scatter_input = EF.dropout(x_neighbor, p=dropout, shared_axes=(-1,), rescale=False,
                                           fill_value=0.0) if dropout > 0.0 else x_neighbor
                x_aggregated = scatter(scatter_input, scatter_index, dim=0, dim_size=node_num, reduce='mean')
            elif aggr_method == 'sum':  # aggr_method == 'sum'
                scatter_input = EF.dropout(x_neighbor, p=dropout, shared_axes=(-1,), rescale=False,
                                           fill_value=0.0) if dropout > 0.0 else x_neighbor
                x_aggregated = scatter(scatter_input, scatter_index, dim=0, dim_size=node_num, reduce='sum')
            elif aggr_method == 'att':
                scatter_input = EF.dropout(x_neighbor, p=dropout, shared_axes=(-1,), rescale=False,
                                           fill_value=0.0) if dropout > 0.0 else x_neighbor
                query = scatter_input.view(edge_num, self.head_num, -1)  # (N, D) -> (N, heads, out_channels)
                if self.att_mode == 'combo':
                    x_center = x[edges[1, :]].view(edge_num, self.head_num, -1)
                    query = torch.cat([query, x_center], dim=-1)  # (N, heads, 2*out_channels)
                alpha = query * self.att_weight
                alpha = alpha.sum(dim=-1)  # (N, heads)
                alpha = F.leaky_relu(alpha, 0.2)  # (N, heads), use leaky relu as in GAT paper
                alpha = scatter_softmax(alpha, scatter_index.view(edge_num, 1), dim=0)
                scatter_input = scatter_input.view(edge_num, self.head_num, -1) * alpha.view(-1, self.head_num,
                                                                                             1)  # (N, heads, out_channels)
                scatter_input = scatter_input.view(edge_num, -1)
                x_aggregated = scatter(scatter_input, scatter_index, dim=0, dim_size=node_num, reduce='sum')
            else:
                raise ValueError('aggregation method = %s not supported' % aggr_method)
            if self.affine_before_merge:
                x_aggregated = self.affine_transforms[i](x_aggregated)
            aggr_outputs.append(x_aggregated)
        if self.multiple_aggregation_merge_method == 'sum':
            x_aggregated = 0
            for i, aggr_out in enumerate(aggr_outputs):
                x_aggregated += aggr_out
        else:  # concatenation
            if len(self.aggregation_methods) > 1:
                x_aggregated = torch.cat(aggr_outputs, dim=-1)
                x_aggregated = self.merge_layer(x_aggregated)  # for dimension normalization
            else:
                x_aggregated = aggr_outputs[0]

        # ---- center update ---#
        if self.update_method == 'sum':
            x += x_aggregated
        elif self.update_method == 'cat':
            x = torch.cat([x, x_aggregated], dim=-1)
        elif self.update_method == 'rnn':
            x = self.rnn(x, x_aggregated)
        elif self.update_method == 'max':
            x = torch.max(torch.stack([x, x_aggregated]), dim=0)[0]
        else:
            raise ValueError('update method = %s not supported' % self.update_method)

        if self.backbone is not None:
            x = self.backbone(x)  # (node_num, output_dim)

        return x


class GraphReadout(nn.Module):
    """
    A generic graph readout op, supported method = {'sum', 'mean', 'max', 'min', 'att', 'rnn-<op1>-<op2>'}, in which
    <op1> & <op2> can be any among {'sum', 'mean', 'max', 'min'}. Degree-wise support removed.
    :param kwargs:  1) att_mode: {'single', 'combo'}, when op = 'att', specify attention mode. Default = 'single', The `att`
                       method is basically correlating node features with the attention vector, this correlation can
                       be done at single node level or at neighbor-center combination level. For the latter mode, attention
                       is done on concatenation of each tuple of (neighbor, center) node features. The center here is a
                       pseudo center constructed by summing all the node features in each graph.
                    2) head_num: default = 1, attention head number
    """

    def __init__(self,
                 input_dim=None,
                 readout_methods=('max', 'sum'),
                 multiple_readout_merge_method='cat',  # {'cat', 'sum'}
                 affine_before_merge=False,
                 **kwargs,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.readout_methods = readout_methods
        self.multiple_readout_merge_method = multiple_readout_merge_method.lower()
        self.affine_before_merge = affine_before_merge
        self.rnns = nn.ModuleList()
        self.att_weights = nn.ParameterList()
        self.affine_transforms = nn.ModuleList()

        for readout_method in readout_methods:
            readout_method = readout_method.lower()
            if readout_method not in {'sum', 'mean', 'max', 'min', 'att'} and not readout_method.startswith('rnn'):
                raise ValueError(
                    "readout_method should be in {'sum', 'mean', 'max', 'min', 'att' or 'rnn-<op1>-<op2>'} but got %s" % readout_method)

            if readout_method.startswith('rnn'):
                assert input_dim is not None, 'input_dim must be specified for `rnn-...` read out method'
                self.rnns.append(nn.GRUCell(input_size=input_dim, hidden_size=input_dim))
            elif readout_method == 'att':
                assert input_dim is not None, 'input_dim must be specified for `att` read out method'
                if 'head_num' in kwargs:
                    self.head_num = kwargs['head_num']
                else:
                    self.head_num = 1
                assert input_dim % self.head_num == 0, 'input_dim must be multiple of head_num'
                if 'att_mode' in kwargs:
                    self.att_mode = kwargs['att_mode']
                else:
                    self.att_mode = 'single'
                assert self.att_mode in {'single', 'combo'}
                if self.att_mode == 'single':
                    self.att_weights.append(Parameter(torch.empty(size=(1, self.head_num, input_dim // self.head_num))))
                else:
                    self.att_weights.append(
                        Parameter(torch.empty(size=(1, self.head_num, 2 * input_dim // self.head_num))))

        feature_dim = input_dim
        self.output_dim = feature_dim
        readout_method_num = len(self.readout_methods)
        if readout_method_num > 1:
            if self.multiple_readout_merge_method == 'cat':
                self.merge_layer = nn.Linear(in_features=feature_dim * readout_method_num, out_features=feature_dim)
        if self.affine_before_merge:
            for i in range(readout_method_num):
                self.affine_transforms.append(nn.Linear(in_features=feature_dim, out_features=feature_dim))
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'att_weights'):
            for w in self.att_weights:
                nn.init.xavier_normal_(w)
        for module in self.rnns:
            module.reset_parameters()
        for module in self.affine_transforms:
            module.reset_parameters()
        if hasattr(self, 'merge_layer'):
            self.merge_layer.reset_parameters()

    def forward(self, x, membership, **kwargs):
        """
        :param x: (node_num, feature_dim)
        :param membership: (node_num,), int64, representing to which graph the i_th node belongs, can be numpy array, no gradient required
        :param kwargs: 1) dropout: dropout value in range [0, 1)
        :return x_readout: (B, feature_dim) if self.degree_wise = False, else (B, feature_dim * (self.max_degree + 1))
        """
        node_num, feature_dim = x.shape
        B = torch.max(membership) + 1  # batch size
        dropout = 0.0
        if 'dropout' in kwargs:
            dropout = kwargs['dropout']
        x_readout_list = []
        for i, readout_method in enumerate(self.readout_methods):
            if readout_method == 'att':
                att_weight = self.att_weights[i]
            elif readout_method.startswith('rnn'):
                rnn = self.rnns[i]

            if readout_method.startswith('rnn'):
                op_center, op_neighbor = readout_method.split('-')[1:]
                h = F.gelu(x)
                pseudo_nodes = []
                for op in [op_center, op_neighbor]:
                    if op == 'mean':
                        h = EF.dropout(h, p=dropout, shared_axes=(1,), rescale=True,
                                       fill_value=0.0) if dropout > 0.0 else h
                        pseudo_node = scatter(h, membership, dim=0, dim_size=B, reduce='mean')
                    elif op == 'max':
                        h = EF.dropout(h, p=dropout, shared_axes=(1,), rescale=False,
                                       fill_value=-1e10) if dropout > 0.0 else h
                        pseudo_node = scatter(h, membership, dim=0, dim_size=B, reduce='max')
                    elif op == 'min':
                        h = EF.dropout(h, p=dropout, shared_axes=(1,), rescale=False,
                                       fill_value=1e10) if dropout > 0.0 else h
                        pseudo_node = scatter(h, membership, dim=0, dim_size=B, reduce='min')
                    else:  # op_neighbor ='sum'
                        h = EF.dropout(h, p=dropout, shared_axes=(1,), rescale=True,
                                       fill_value=0.0) if dropout > 0.0 else h
                        pseudo_node = scatter(h, membership, dim=0, dim_size=B, reduce='sum')
                    pseudo_nodes.append(pseudo_node)
                x_readout = rnn(pseudo_nodes[0], pseudo_nodes[1])
            elif readout_method == 'att':
                x_neighbor = EF.dropout(x, p=dropout, shared_axes=(1,), rescale=True,
                                        fill_value=0.0) if dropout > 0.0 else x
                x_neighbor = x_neighbor.view(node_num, self.head_num, -1)  # (N, D) -> (N, heads, out_channels)
                query = x_neighbor
                if self.att_mode == 'combo':
                    x_center = scatter(x, membership, dim=0, dim_size=B, reduce='sum')[membership, :]
                    x_center = x_center.view(node_num, self.head_num, -1)
                    query = torch.cat([query, x_center], dim=-1)  # (N, heads, 2*out_channels)
                alpha = query * att_weight
                alpha = alpha.sum(dim=-1)  # (N, heads)
                alpha = F.leaky_relu(alpha, 0.2)  # (N, heads), use leaky relu as in GAT paper
                alpha = scatter_softmax(alpha, membership.view(node_num, 1), dim=0)
                x_neighbor = x_neighbor * alpha.view(-1, self.head_num, 1)  # (N, heads, out_channels)
                x_neighbor = x_neighbor.view(node_num, -1)
                x_readout = scatter(x_neighbor, membership, dim=0, dim_size=B, reduce='sum')
            elif readout_method == 'mean':
                scatter_input = EF.dropout(x, p=dropout, shared_axes=(1,), rescale=True,
                                           fill_value=0.0) if dropout > 0.0 else x
                x_readout = scatter(scatter_input, membership, dim=0, dim_size=B, reduce='mean')
            elif readout_method == 'max':
                scatter_input = EF.dropout(x, p=dropout, shared_axes=(1,), rescale=False,
                                           fill_value=-1e10) if dropout > 0.0 else x
                x_readout = scatter(scatter_input, membership, dim=0, dim_size=B, reduce='max')
            elif readout_method == 'min':
                scatter_input = EF.dropout(x, p=dropout, shared_axes=(1,), rescale=False,
                                           fill_value=1e10) if dropout > 0.0 else x
                x_readout = scatter(scatter_input, membership, dim=0, dim_size=B, reduce='min')
            else:  # 'sum'
                scatter_input = EF.dropout(x, p=dropout, shared_axes=(1,), rescale=True,
                                           fill_value=0.0) if dropout > 0.0 else x
                x_readout = scatter(scatter_input, membership, dim=0, dim_size=B, reduce='sum')

            if self.affine_before_merge:
                x_readout = self.affine_transforms[i](x_readout)
            x_readout_list.append(x_readout)

        if self.multiple_readout_merge_method == 'sum':
            x_readout = 0
            for i, out in enumerate(x_readout_list):
                x_readout += out
        else:  # concatenation
            if len(self.readout_methods) > 1:
                x_readout = torch.cat(x_readout_list, dim=-1)
                x_readout = self.merge_layer(x_readout)  # for dimension normalization
            else:
                x_readout = x_readout_list[0]

        return x_readout


if __name__ == '__main__':
    INFO = ['This is code repo for generic graph operations\n',
            'Author: David Leon\n',
            'All rights reserved\n']
    print(*INFO)
