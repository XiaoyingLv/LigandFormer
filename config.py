# coding:utf-8
"""
Parameter configurations for different experiments
Created  :   7, 11, 2019
Revised  :   7, 11, 2019
Author   :  David Leon (dawei.leng@ghddi.org)
All rights reserved
-------------------------------------------------------------------------------
"""
__author__ = 'dawei.leng'


class CONFIG(object):
    def __init__(self):
        super().__init__()
        self.version = None


class model_4v4_config(CONFIG):
    def __init__(self):
        super().__init__()
        self.version = '1.0.1, 10-13-2020'
        self.block_num = 5
        self.input_dim = 75
        self.hidden_dim = 256
        self.degree_wise = False
        self.max_degree = 26
        self.aggregation_methods = ['max', 'sum']
        self.multiple_aggregation_merge_method = 'sum'
        self.multiple_readout_merge_method = 'cat'
        self.affine_before_merge = False
        self.node_feature_update_method = 'cat'
        self.readout_methods = ['max']
        self.add_dense_connection = False
        self.pyramid_feature = True
        self.slim = True
        self.norm_method = 'ln'  # 'bn', 'ln', 'wn', 'bn_notrack', 'none'
        self.classifier_dim = self.hidden_dim


model_4v4_1_config = model_4v4_config
model_4v4_multifeats_config = model_4v4_config

class model_4v7_config(CONFIG):
    def __init__(self):
        super().__init__()
        self.version = '1.0.1, 10-23-2020'
        self.block_num = 5
        self.input_dim = 75
        self.hidden_dim = 256
        self.degree_wise = False
        self.max_degree = 26
        self.aggregation_methods = ['max', 'sum']
        self.multiple_aggregation_merge_method = 'sum'
        self.multiple_readout_merge_method = 'cat'
        self.affine_before_merge = False
        self.node_feature_update_method = 'rnn'
        self.readout_methods = ['max']
        self.norm_method = 'newbn'  # 'bn', 'ln', 'wn', 'bn_notrack', 'none'


model_4v7_1_config = model_4v7_config


class model_4v5_config(CONFIG):
    def __init__(self):
        super().__init__()
        self.input_dim = 75
        self.degree_wise = True
        self.max_degree = 26
        self.affine_before_merge = False
        self.readout_methods = ['mean', 'max']
        self.multiple_readout_merge_method = 'cat'


model_4v5_1_config = model_4v5_config


class model_4v6_config(CONFIG):
    def __init__(self):
        super().__init__()
        self.input_dim = 75
        # self.hidden_dims = (256, 64, 32)
        # self.hidden_dims = (1024, 512, 256, 128)
        self.hidden_dims = (256, 512, 512, 256, 128)
        self.aggregation_methods = ('max', 'mean')
        self.multiple_aggregation_merge_method = 'sum'
        self.affine_before_merge = False
        self.node_feature_update_method = 'rnn'
        self.readout_methods = ('rnn-sum-max',)
        self.multiple_readout_merge_method = 'sum'
        self.pyramid_feature = False
        self.degree_wise = False
        # self.dropouts = [(0.5, 0), (0.7, 100), (0.5, 500), (0.0, 700)]
        # self.anneal_dropouts = [(0.5, 0), (0.7, 0.1), (0.3, 0.2), (0.2, 0.3), (0.0, 0.4)]
        # self.anneal_dropouts = [(0.3, 0), (0.2, 0.1), (0.1, 0.2), (0.1, 0.4), (0.0, 0.5)]


model_4v6_1_config = model_4v6_config


# ------------- obsolete configs ---------------------#
class model_2_config(CONFIG):
    def __init__(self):
        super().__init__()
        self.block_num = 4
        self.block_layer_num = 2
        self.input_dim = 75
        self.hidden_dim = 64
        self.final_dropout = 0
        self.neighbor_pooling_method = 'sum'
        self.readout_method = 'sum'


class model_2_1_config(CONFIG):
    def __init__(self):
        super().__init__()
        self.block_num = 4
        self.block_layer_num = 2
        self.input_dim = 75
        self.hidden_dim = 64
        self.final_dropout = 0
        self.neighbor_pooling_method = 'sum'
        self.readout_method = 'sum'


class model_2_max_config(CONFIG):
    def __init__(self):
        super().__init__()
        self.block_num = 4
        self.block_layer_num = 2
        self.input_dim = 75
        self.hidden_dim = 64
        self.final_dropout = 0.1
        self.neighbor_pooling_method = 'max'
        self.readout_method = 'sum'


class model_2_1_hidden128_config(CONFIG):
    def __init__(self):
        super().__init__()
        self.block_num = 4
        self.block_layer_num = 2
        self.input_dim = 75
        self.hidden_dim = 128
        self.final_dropout = 0.1
        self.neighbor_pooling_method = 'sum'
        self.readout_method = 'sum'


class model_2_1_weighting_config(CONFIG):
    def __init__(self):
        super().__init__()
        self.block_num = 4
        self.block_layer_num = 2
        self.input_dim = 75
        self.hidden_dim = 64
        self.final_dropout = 0.1
        self.neighbor_pooling_method = 'sum'
        self.readout_method = 'sum'
        self.class_weight = [1.0, 5.0]


class model_3_config(CONFIG):
    def __init__(self):
        super().__init__()
        self.ks = (0.9, 0.7, 0.6, 0.5)
        self.input_dim = 75
        self.hidden_dim = 64
        self.dropout = 0
        self.readout_method = 'sum'


class model_3_1_config(CONFIG):
    def __init__(self):
        super().__init__()
        self.ks = (0.9, 0.7, 0.6, 0.5)
        self.input_dim = 75
        self.hidden_dim = 64
        self.dropout = 0
        self.readout_method = 'sum'


class model_3v1_config(CONFIG):
    def __init__(self):
        super().__init__()
        self.ks = (0.9, 0.7, 0.6, 0.5)
        self.input_dim = 75
        self.hidden_dim = 64
        self.dropout = 0.1
        self.readout_method = 'sum'


class model_3v1_1_config(CONFIG):
    def __init__(self):
        super().__init__()
        self.ks = (0.9, 0.7, 0.6, 0.5)
        self.input_dim = 75
        self.hidden_dim = 64
        self.dropout = 0.1
        self.readout_method = 'sum'


class model_4_config(CONFIG):
    def __init__(self):
        super().__init__()
        self.block_num = 5
        self.input_dim = 75
        self.hidden_dim = 256
        self.dropout = 0.5
        self.aggregation_method = 'sum_max'
        self.readout_method = 'sum'
        self.eps = 0.1
        self.add_dense_connection = True


class model_4v1_config(CONFIG):
    def __init__(self):
        super().__init__()
        self.block_num = 5
        self.input_dim = 75
        self.hidden_dim = 256
        self.aggregation_methods = ['rnn', 'sum']
        self.affine_before_merge = False
        self.multiple_aggregation_merge = 'cat'
        self.readout_method = 'sum'
        self.eps = 0.1
        self.add_dense_connection = True
        # self.dropouts = [(0.5, 0), (0.7, 100), (0.5, 500), (0.0, 700)]
        # self.anneal_dropouts = [(0.5, 0), (0.7, 0.1), (0.3, 0.2), (0.2, 0.3), (0.0, 0.4)]
        # self.anneal_dropouts = [(0.3, 0), (0.2, 0.1), (0.1, 0.2), (0.1, 0.4), (0.0, 0.5)]


model_4_1_config = model_4_config
model_4_1_config.input_dim = 75
model_4v1_1_config = model_4v1_config
model_4v1_1_config.input_dim = 75


class model_4v2_config(CONFIG):
    def __init__(self):
        super().__init__()
        self.block_num = 5
        self.input_dim = 75
        self.hidden_dim = 512
        self.readout_method = 'sum'
        self.add_dense_connection = True


model_4v2_1_config = model_4v2_config


class model_4v3_config(CONFIG):
    def __init__(self):
        super().__init__()
        self.block_num = 5
        self.input_dim = 75
        self.hidden_dim = 256
        self.aggregation_methods = ['sum', 'rnn-max']
        self.affine_before_merge = False
        self.multiple_aggregation_merge = 'cat'
        self.readout_method = 'rnn-sum-max'
        self.eps = 1.0
        self.add_dense_connection = True
        self.use_neighbor_op = False
        # self.dropouts = [(0.5, 0), (0.7, 100), (0.5, 500), (0.0, 700)]
        # self.anneal_dropouts = [(0.5, 0), (0.7, 0.1), (0.3, 0.2), (0.2, 0.3), (0.0, 0.4)]
        # self.anneal_dropouts = [(0.3, 0), (0.2, 0.1), (0.1, 0.2), (0.1, 0.4), (0.0, 0.5)]


model_4v3_1_config = model_4v3_config


class model_0_config(CONFIG):
    def __init__(self):
        super().__init__()
        self.feature_dim = 75
        self.graph_conv_layer_size = (64, 64)
        self.dense_layer_size = 128
        self.dropout = 0
        self.min_degree = 0
        self.max_degree = 20


class model_0_0_config(CONFIG):
    """
    For model_0 with embedding input
    """

    def __init__(self):
        super().__init__()
        self.feature_dim = 75
        self.graph_conv_layer_size = (64, 64)
        self.dense_layer_size = 128
        self.dropout = 0
        self.min_degree = 0
        self.max_degree = 50


class model_1_config(CONFIG):
    def __init__(self):
        super().__init__()
        self.feature_dim = 75
        self.hidden_size = 128
        self.bidirectional = True
        self.dropout = 0


class model_1_1_config(CONFIG):
    def __init__(self):
        super().__init__()
        self.feature_dim = 75
        self.hidden_size = 128
        self.bidirectional = True
        self.dropout = 0


class model_4v4_config0(CONFIG):
    def __init__(self):
        super().__init__()

        self.block_num = 10
        self.input_dim = 75
        self.hidden_dim = 256
        self.degree_wise = False
        self.max_degree = 26
        self.aggregation_methods = ['max', 'mean']
        self.multiple_aggregation_merge_method = 'sum'
        self.multiple_readout_merge_method = 'sum'
        self.affine_before_merge = False
        self.node_feature_update_method = 'rnn'
        self.readout_methods = ['rnn-max-sum']
        self.add_dense_connection = True  # whether add dense connection among the blocks
        self.pyramid_feature = True
        self.slim = True
        # self.dropouts = [(0.5, 0), (0.7, 100), (0.5, 500), (0.0, 700)]
        # self.anneal_dropouts = [(0.5, 0), (0.7, 0.1), (0.3, 0.2), (0.2, 0.3), (0.0, 0.4)]
        # self.anneal_dropouts = [(0.3, 0), (0.2, 0.1), (0.1, 0.2), (0.1, 0.4), (0.0, 0.5)]
        self.norm_method = 'ln'  # 'bn', 'ln', 'wn', 'bn_notrack', 'none'
        self.classifier_dim = self.input_dim  # self.hidden_dim
# ---------- end of obsolete configs -----------------#
