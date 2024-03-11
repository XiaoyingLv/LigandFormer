# coding:utf-8
'''
Virtual screening based on classification of compound with ligand properties
Created  :   6, 11, 2019
Revised  :   6, 12, 2019   add balance_resample() and dataset_shuffle() functions
            11, 28, 2019   add support for regression task
Author   :  David Leon (dawei.leng@ghddi.org)
            Pascal Guo (jinjiang.guo@ghddi.org)
All rights reserved
'''
# ------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng, jinjiang.guo'

import copy
import json
import os
import time, warnings, shutil, scipy as sp, scipy.stats
import multiprocessing
import queue
import random

if os.name == 'posix':
    import matplotlib
    matplotlib.use('svg')
import torch
from torch.optim import Adadelta, Adam, SGD
# from torch.optim.lr_scheduler import LambdaLR
from torch.nn.functional import softmax
from pytorch_ext.util import get_local_ip, get_time_stamp, finite_memory_array, gpickle, sys_output_tap, verbose_print
from pytorch_ext.util import freeze_module, unfreeze_module, get_trainable_parameters, set_value
from ligand_based_VS_data_preprocessing import calc_class_distribution, balance_resample, dataset_partition_by_class, \
    dataset_partition_for_regress, dataset_shuffle, get_scaffold_group
from ligand_based_VS_evaluation import compute_confusion_matrix
import config as experiment_configs

import metrics
import data_loader
from graph_utils import *
from util.convergence_plot import plot_convergence_curve
from util.mfold_variance_plot import plot_m_variance

from util.regression_convergence_plot import plot_regress_convergence_curve
from util.scaffold_overlap import scaffold_overlap_summary, scaffold_summary
from ligand_based_VS_model import Model_Agent
from util.mfold_selection import split_y

torch.set_num_threads(2)

try:
    local_ip = get_local_ip()
except:
    local_ip = 'None'


def init_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def whiten_norm(x):
    x = x - np.mean(x, axis=(1, 2, 3), keepdims=True)
    x = x / (np.mean(x ** 2, axis=(1, 2, 3), keepdims=True) ** 0.5)
    return x


def clear_cuda_cache(model):
    for p in model.parameters():
        if p.grad is not None:
            del p.grad  # free some memory
    torch.cuda.empty_cache()


def compute_alpha_enrichment(y_true, y_score, alpha):
    """
    compute alpha enrichment factors, only for binary classification task
    :param y_true: 1D np array
    :param y_score: 1D np array, predicted scores for positive class
    :param alpha: list/tuple of float in (0, 1.0]
    :return: alpha_acc_list, relative_enrichment_list. For their explanations refer to `metrics.alpha_enrichment()`
    """
    alpha_acc_list, relative_enrichment_list, positive_ratio = [], [], None
    for a in alpha:
        alpha_acc, positive_ratio, relative_enrichment = metrics.alpha_enrichment(y_true, y_score, alpha=a)
        alpha_acc_list.append(alpha_acc)
        relative_enrichment_list.append(relative_enrichment)
    return alpha_acc_list, relative_enrichment_list, positive_ratio


def compute_score_level_enrichment(y_true, y_score, scores):
    """
    compute score level relative enrichment, only for binary classification task
    :param y_true: 1D np array
    :param y_score: 1D np array, predicted scores for positive class
    :param scores: list/tuple of float in (0, 1.0)
    :return: level_acc_list, relative_enrichment_list. For their explanations refer to `metrics.score_level_enrichment()`
    """
    level_acc_list, relative_enrichment_list, positive_ratio = [], [], None
    for score in scores:
        level_acc, positive_ratio, relative_enrichment = metrics.score_level_enrichment(y_true, y_score, score=score)
        level_acc_list.append(level_acc)
        relative_enrichment_list.append(relative_enrichment)
    return level_acc_list, relative_enrichment_list, positive_ratio


def get_annealing_dropout(current_epoch, anneal_dropouts, max_epoch=None):
    """
    Dropout annealing. The annealing curve is piece-wise linear, defined by `anneal_dropouts`.
    The `anneal_dropouts` is a (n, 2) shaped np.ndarray or an equal list, each row of `anneal_dropouts` is of
    format (dropout, idx), in which idx is either integer (unscaled) or float <= 1.0 (scaled), in the latter case
    the `max_epoch` must be specified.
    :param current_epoch:
    :param anneal_dropouts:
    :param max_epoch:
    :return:
    """
    if isinstance(anneal_dropouts, list):
        anneal_dropouts = np.array(anneal_dropouts)
    if np.all(anneal_dropouts[:, 1] <= 1.0):
        if max_epoch is None:
            raise ValueError('max_epoch must be specified if scaled anneal_dropouts is used')
        anneal_dropouts[:, 1] *= max_epoch
    n = anneal_dropouts.shape[0]
    idx = n
    for i in range(n):
        if current_epoch < anneal_dropouts[i, 1]:
            idx = i
            break
    if idx == n:
        dropout = anneal_dropouts[-1, 0]
    else:
        p1, p2 = anneal_dropouts[idx - 1, 0], anneal_dropouts[idx, 0]
        x1, x2 = anneal_dropouts[idx - 1, 1], anneal_dropouts[idx, 1]
        x = current_epoch
        dropout = (x - x1) / (x2 - x1) * (p2 - p1) + p1
    return dropout


def _retrieve_dataset_from_save_folder(save_folder):
    src_folder = [x.name for x in os.scandir(save_folder) if x.is_dir() and x.name.startswith('src_')]
    assert len(src_folder) == 1, 'Exception: total %d src folders located under %s' % (len(src_folder), save_folder)
    src_folder = os.path.join(save_folder, src_folder[0])
    trainset_files = [x.name for x in os.scandir(src_folder) if x.is_file() and x.name.startswith('trainset@')]
    assert len(trainset_files) == 1, 'Exception: total %d trainset files located under %s' % (
    len(trainset_files), src_folder)
    testset_files = [x.name for x in os.scandir(src_folder) if x.is_file() and x.name.startswith('testset@')]
    assert len(testset_files) == 1, 'Exception: total %d testset files located under %s' % (
    len(testset_files), src_folder)
    trainset_file = os.path.join(src_folder, trainset_files[0])
    testset_file = os.path.join(src_folder, testset_files[0])
    return trainset_file, testset_file


class TrainLog(object):

    def __init__(self):
        super().__init__()
        self.save_folder = None
        self.time_stamp = None
        self.local_ip = None
        self.log_file = None
        self.best_aupr = None
        self.best_ER_test = None
        self.best_corr = None
        self.best_rmse_test = None
        self.best_ER_train = None
        self.best_rmse_train = None
        self.best_model = None
        self.best_tmp_model = None
        self.epoch = None


def train(args):
    # --- public paras ---#
    model_file = args.model_file
    trainset = args.trainset  # file path or tuple in format of (#todo)
    testset = args.testset  # file path or tuple in format of (#todo) or float in (0, 1.0)
    batch_size = args.batch_size
    batch_size_min = args.batch_size_min  # if batch_size_min < batch_size, during training, batch size randomization will be enabled
    batch_size_test = args.batch_size_test
    save_root_folder = args.save_root_folder
    save_folder = args.save_folder  # set this if you want resume from last run
    balance_method = args.balance_method
    weighting_method = args.weighting_method
    prefix = args.prefix
    config_set = args.config
    use_pretrained = args.pretrained
    update_per_batch = args.upb
    accum_loss_num = args.accum_loss_num
    time_stamp = get_time_stamp()
    trainlog = TrainLog()

    if not isinstance(config_set, experiment_configs.CONFIG):
        if config_set is None:
            config_set = 'model_%s_config' % args.model_ver
        config = getattr(experiment_configs, config_set)()
    else:
        config = config_set  # already instance of CONFIG class

    if prefix is None:
        prefix = 'model_%s' % args.model_ver
    vprint = verbose_print(level=args.verbose, prefix=prefix)
    stdout_tap = sys_output_tap(sys.stdout, only_output_to_file=args.silent)
    stderr_tap = sys_output_tap(sys.stderr, only_output_to_file=args.silent)
    sys.stdout = stdout_tap
    sys.stderr = stderr_tap

    if save_root_folder is None:
        save_root_folder = os.path.join(os.getcwd(), 'train_results')
    if not os.path.exists(save_root_folder):
        os.makedirs(save_root_folder)
    if save_folder is None:
        save_folder = os.path.join(save_root_folder,
                                   '[%s]_VS_%s_model_%s@%s' % (prefix, args.task, args.model_ver, time_stamp))
    model_folder = os.path.join(save_folder, 'model')
    backup_folder = os.path.join(save_folder, 'src_model_%s@%s' % (args.model_ver, time_stamp))
    stdout_log_file = os.path.join(save_folder, '[%s]_VS_%s_model_%s@%s_stdout.txt' % (
    prefix, args.task, args.model_ver, time_stamp,))
    stderr_log_file = os.path.join(save_folder, '[%s]_VS_%s_model_%s@%s_stderr.txt' % (
    prefix, args.task, args.model_ver, time_stamp,))
    logfile = os.path.join(save_folder, '[%s]_VS_%s_model_%s@%s.log' % (prefix, args.task, args.model_ver, time_stamp))
    epoch_start = 0
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        resume_run = False
    else:  # try to resume from last run
        resume_run = True
        try:
            logfile = [os.path.join(save_folder, x) for x in os.listdir(save_folder) if x.endswith(".log")][0]
            # ---- retrieve last epoch ----#
            with open(logfile, mode='rt', encoding='utf8') as f:
                for line in f:
                    line = line.rstrip()
                    if len(line) > 0:
                        if line.startswith('epoch =') and 'done' in line:
                            epoch_start = int(line.split(',')[0][7:-4])
            # ---- retrieve last model ----#
            model_files = [os.path.join(model_folder, x) for x in os.listdir(model_folder) if
                           x.endswith(".pkl") or x.endswith(".pt")]
            model_file_newest = max(model_files, key=os.path.getctime)
            if model_file is not None and resume_run:
                warnings.warn('Resume run: input model file will be ignored = %s' % model_file)
            model_file = model_file_newest
            # ---- other setup ----#
            stdout_log_file = os.path.basename(save_folder) + '_stdout.log'
            stderr_log_file = os.path.basename(save_folder) + '_stderr.log'
        except:
            resume_run = False
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    stdout_tap_log = open(stdout_log_file, 'at')
    stderr_tap_log = open(stderr_log_file, 'at')
    stdout_tap.set_file(stdout_tap_log)
    stderr_tap.set_file(stderr_tap_log)
    if resume_run:
        vprint('Resume from last run ...', l=1)
        vprint('  model_file = %s' % model_file, l=1)
        vprint('  epoch_start = %d' % epoch_start)
    vprint('============== args ====================', l=1)
    for key in args.__dict__:
        print(key, '=', args.__dict__[key])
    vprint('========================================\n', l=1)
    trainlog.save_folder = save_folder
    trainlog.time_stamp = time_stamp
    trainlog.local_ip = local_ip

    if args.embedding_file is not None:
        try:
            embedding_weight, source_model_path = gpickle.load(args.embedding_file)
        except:
            args.embedding_file = None
            warnings.warn('Failed for loading embedding file = %s', args.embedding_file)

    # --- (0) parameter setup ---#
    class_weight = None
    if hasattr(args, 'class_weight'):
        class_weight = np.array(args.class_weight)
        class_weight /= class_weight.sum()
        weighting_method = 0

    if weighting_method > 0:
        balance_method = 0

    # --- (1) list source files to backup ---#
    files_to_backup = ['ligand_based_VS_train.py',
                       'ligand_based_VS_model.py',
                       'ligand_based_VS_data_preprocessing.py',
                       'ligand_based_VS_model_experimental.py',
                       'ligand_based_VS_model_obsolete.py',
                       'ligand_based_VS_evaluation.py',
                       'config.py',
                       'metrics.py',
                       'data_loader.py',
                       'graph_utils.py',
                       'graph_ops.py',
                       'grid_search.py',
                       'featurizer.py',
                       'readme.md',
                       'changes.md',
                       'util/convergence_plot.py',
                       'util/regression_convergence_plot.py',
                       'util/export_grid_search_history.py',
                       'util/evaluation_with_csv.py',
                       ]
    if args.embedding_file is not None:
        files_to_backup.append(args.embedding_file)

    # --- (2) setup device ---#
    substrs = args.device.split(',')
    device = []
    for i, substr in enumerate(substrs):
        dv = int(substr)
        if dv < 0:
            device.append(torch.device('cpu'))
        else:
            device.append(torch.device('cuda:%d' % dv))
        print('Device[%d] = ' % i, device[-1])
    if len(device) == 1:
        device = device[0]

    # --- (3) initialize model ---#
    vprint('Initialize model agent', l=1)
    if args.model_ver in {'1'}:
        atom_dict_file = 'Mtb_mt_charset_canonicalized.gpkl'
    else:
        atom_dict_file = 'atom_dict.gpkl'
    vprint('config = ', l=1)
    vprint(config.__dict__, l=1)
    model_agent = Model_Agent(device=device, model_ver=args.model_ver, output_dim=args.class_num, task=args.task,
                              config=config, model_file=model_file, atom_dict_file=atom_dict_file,
                              load_weights_strict=args.load_weights_strict)
    if model_agent.model_file_md5 is not None:
        vprint('model weights loaded', l=1)
    if args.reset_classifier:
        for module in (model_agent.model.dense0, model_agent.model.bn0, model_agent.model.dense1, model_agent.model.bn1,
                       model_agent.model.dense2):
            module.reset_parameters()
    # if args.embedding_file is not None:
    #     if hasattr(model_agent.model, 'emb0') and args.model_ver not in {'1', '1_1'}:
    #         set_value(model_agent.model.emb0.weight, embedding_weight[:-1,:])
    #         # freeze_module(model_agent.model.emb0)
    #         vprint('embedding weight loaded', l=1)

    if args.model_ver == '4v6' and use_pretrained > 0:
        # GE_model_0_file = r"GE_model_0@2020-01-03_09-45-06_batch=2130000_PPL=[1.08,1.10].pkl"
        GE_model_0_file = r"tmp_GE_model_0@2020-01-03_09-46-12_batch=1642000_PPL=1.12.pkl"
        state_dict = torch.load(GE_model_0_file, map_location=device)
        model_agent.model.load_state_dict(state_dict, strict=False)
        set_value(model_agent.model.emb0.weight, state_dict['embedding.weight'].cpu().numpy()[:-1, :])
        vprint('GE weights loaded for model_4v6', l=1)

    vprint('Model version = %s' % args.model_ver, l=1)
    vprint('Configuration set = %s' % config_set, l=1)
    # model = torch.jit.script(model)
    # for name, p in model.named_parameters():
    # vprint(['p.device=', name, torch.get_device(p)])

    # --- (4) initialize optimizer ---#
    weight_decay = args.weight_decay if hasattr(args, 'weight_decay') else 0

    # todo: this section should be unnecessary, just for debugging
    # if use_pretrained > 0:
    #     if args.model_ver == '4v4':
    #         for module in [model_agent.model.emb0, model_agent.model.blocks, model_agent.model.readout_ops]:
    #             freeze_module(module)
    #         vprint('CL pretrained modules freezed', l=1)
    #     elif args.model_ver == '4v6':
    #         for module in [model_agent.model.emb0, model_agent.model.blocks]:
    #             freeze_module(module)
    #         vprint('4v6 pretrained modules freezed', l=1)

    params_to_train = get_trainable_parameters(model_agent.model, with_name=False)
    # for param in params_to_train:
    #     print('param_to_train:', param[0])
    if args.optimizer.lower() == 'adadelta':
        lr = args.lr if args.lr is not None else 1.0
        optimizer = Adadelta(params_to_train, weight_decay=weight_decay, lr=lr)
    elif args.optimizer.lower() == 'adam':
        lr = args.lr if args.lr is not None else 1e-3
        optimizer = Adam(params_to_train, weight_decay=weight_decay, lr=lr)
    elif args.optimizer.lower() == 'sgd':
        lr = args.lr if args.lr is not None else 1e-2
        if hasattr(args, 'momentum'):
            momentum = args.momentum
        else:
            momentum = 0.1
        optimizer = SGD(params_to_train, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError('optimizer = %s not supported' % args.optimizer)
    optimizer.zero_grad()

    # --- (5) setup data ---#
    if batch_size_test is None:
        batch_size_test = batch_size

    # ---- (5.1) backup src files ----#
    for file in files_to_backup:
        src_folder = os.path.dirname(__file__)
        full_file_path_src = os.path.join(src_folder, file)
        full_file_path_tgt = os.path.join(backup_folder, os.path.basename(file))
        if os.path.exists(full_file_path_tgt):
            file, ext = os.path.splitext(full_file_path_tgt)
            file += '_%s' % time_stamp
            full_file_path_tgt = file + ext
            warnings.warn('target file exists, new copy renamed to %s' % full_file_path_tgt)
        if not os.path.isfile(full_file_path_src):
            warnings.warn('source file = %s not exist' % file, RuntimeWarning)
        else:
            shutil.copyfile(full_file_path_src, full_file_path_tgt)
            vprint('File = %s --> %s' % (file, backup_folder), l=1)

    vprint('local ip = %s, model ver = %s, time_stamp = %s' % (local_ip, args.model_ver, time_stamp), l=1)

    # ---- (5.2) load samples ----#
    if args.srcfolder is not None:
        if args.trainset is not None or args.testset is not None:
            raise ValueError('srcfolder and (trainset, testset) should not be specified simultaneously')
        else:
            trainset, testset = _retrieve_dataset_from_save_folder(args.srcfolder)

    try:
        partition_ratio = float(testset)
    except (ValueError, TypeError):
        partition_ratio = None

    if isinstance(trainset, str):
        trainset = gpickle.load(trainset)

    trainset_backup_file = os.path.join(backup_folder, 'trainset@%s.gpkl' % time_stamp)
    testset_backup_file = os.path.join(backup_folder, 'testset@%s.gpkl' % time_stamp)
    if partition_ratio is not None and trainset is not None:
        assert 0.0 < partition_ratio < 1.0
        if isinstance(trainset, dict):
            features = trainset['features']
            sample_num = features.shape[0]
        else:
            sample_num = trainset[0].shape[0]
        if args.group_by_scaffold:
            scaffold_dict = trainset['scaffold_dict']
            scaffold_dict_keys = list(scaffold_dict)
            np.random.shuffle(scaffold_dict_keys)
            index = []
            for key in scaffold_dict_keys:
                np.random.shuffle(scaffold_dict[key])
                index.extend(scaffold_dict[key])
        else:
            index = np.arange(sample_num)
            np.random.shuffle(index)
        trainset = dataset_shuffle(trainset, index)
        if args.task == 'classification':
            labels = trainset['labels'] if isinstance(trainset, dict) else trainset[1]
            trainset, testset = dataset_partition_by_class(trainset, labels, (1.0 - partition_ratio, partition_ratio))
        elif args.task == 'regression':
            trainset, testset = dataset_partition_for_regress(trainset, (
            1.0 - partition_ratio, partition_ratio))  # todo: add support for dict-typed dataset

        gpickle.dump(trainset, trainset_backup_file)
        gpickle.dump(testset, testset_backup_file)

    else:
        if isinstance(args.trainset, str):
            shutil.copyfile(args.trainset, trainset_backup_file)
        else:
            gpickle.dump(trainset, trainset_backup_file)
        if isinstance(testset, str):
            shutil.copyfile(testset, testset_backup_file)
            testset = gpickle.load(testset)
        else:
            gpickle.dump(testset, testset_backup_file)
    if isinstance(trainset, dict):
        mol_features_train = trainset['features']
        ground_truths_train = trainset['labels']
        SMILES_train = trainset['SMILESs']
        IDs_train = trainset['IDs']
        sample_weights_train = trainset['sample_weights']
    else:
        mol_features_train, ground_truths_train, SMILES_train, IDs_train, *sample_weights_train = trainset
    if isinstance(testset, dict):
        mol_features_test = testset['features']
        ground_truths_test = testset['labels']
        SMILES_test = testset['SMILESs']
        IDs_test = testset['IDs']
    else:
        mol_features_test, ground_truths_test, SMILES_test, IDs_test, *aux_data_list = testset

    if args.task == 'regression':  # regression:
        ground_truths_train = ground_truths_train.astype(np.float32)
        ground_truths_test = ground_truths_test.astype(np.float32)
    else:
        ground_truths_train = ground_truths_train.astype(np.int64)
        ground_truths_test = ground_truths_test.astype(np.int64)

    train_sample_num = mol_features_train.shape[0]
    test_sample_num = mol_features_test.shape[0]
    if sample_weights_train is None or len(sample_weights_train) == 0 or isinstance(sample_weights_train,
                                                                                    list) and None in \
            sample_weights_train[0]:
        sample_weights_train = np.ones(train_sample_num).astype(np.float32)
    else:
        if isinstance(sample_weights_train, list):  # for back-compatibility
            sample_weights_train = sample_weights_train[0]
        sample_weights_train = sample_weights_train.astype(np.float32)
    if args.train_partial < 1.0:
        train_sample_num = max(int(train_sample_num * args.train_partial), 0)
        mol_features_train = mol_features_train[:train_sample_num, ::]
        ground_truths_train = ground_truths_train[:train_sample_num]
        SMILES_train = SMILES_train[:train_sample_num]
        IDs_train = IDs_train[:train_sample_num]
        sample_weights_train = sample_weights_train[:train_sample_num]

    vprint('\ntrain_partial = %0.2f' % args.train_partial, l=1)
    vprint('train samples = %d, test samples = %d' % (train_sample_num, test_sample_num), l=1)

    # ---- (5.3) class weighting ----#
    if weighting_method == 1:
        vprint('Weighting method 1', l=1)
        class_distribution = calc_class_distribution(ground_truths_train, args.class_num)
        class_weight = 1.0 / class_distribution
        class_weight /= class_weight.sum()
    elif weighting_method == 2:
        vprint('Weighting method 2', l=1)
        class_distribution = calc_class_distribution(ground_truths_train, args.class_num)
        class_weight = 1.0 / class_distribution
        class_weight = np.sqrt(class_weight)
        class_weight /= class_weight.sum()
    else:
        if class_weight is None:
            vprint('Uniform class weighting', l=1)
            class_weight = np.ones(args.class_num, dtype=np.float32) / args.class_num
    class_weight = class_weight * args.class_num
    class_weight = class_weight.astype(np.float32)
    vprint('class_weight=', class_weight, l=1)

    # --- (6) create log file ---#
    trainlog.log_file = logfile
    logfp = open(logfile, mode='at', encoding='utf8')
    if not resume_run:
        logfp.write('device                     = %s\n' % args.device)
        logfp.write('time stamp                 = %s\n' % time_stamp)
        logfp.write('local_ip                   = %s\n' % local_ip)
        logfp.write('prefix                     = %s\n' % prefix)
        logfp.write('model version              = %s\n' % args.model_ver)
        logfp.write('model file                 = %s\n' % model_file)
        logfp.write('config                     = %s\n' % config_set)
        logfp.write('class_num                  = %d\n' % args.class_num)
        logfp.write('pred_threshold             = %d\n' % args.pred_threshold)
        if isinstance(args.trainset, str):
            logfp.write('trainset                   = %s\n' % args.trainset)
        if isinstance(args.testset, str):
            logfp.write('testset                    = %s\n' % args.testset)
        if isinstance(args.srcfolder, str):
            logfp.write('srcfolder                  = %s\n' % args.srcfolder)
        logfp.write('train set size             = %d\n' % train_sample_num)
        logfp.write('test  set size             = %d\n' % test_sample_num)
        logfp.write('optimizer                  = %s\n' % args.optimizer)
        logfp.write('batch_size                 = %d\n' % batch_size)
        logfp.write('batch_size_min             = %s\n' % batch_size_min)
        logfp.write('batch_size_test            = %d\n' % batch_size_test)
        logfp.write('max_epoch_num              = %d\n' % args.max_epoch_num)
        logfp.write('save_model_per_epoch_num   = %0.3f\n' % args.save_model_per_epoch_num)
        logfp.write('test_model_per_epoch_num   = %0.3f\n' % args.test_model_per_epoch_num)
        logfp.write('ER_start_test              = %0.1f\n' % args.ER_start_test)
        logfp.write('RMSE_start_test            = %0.1f\n' % args.RMSE_start_test)
        logfp.write('train_loader_worker        = %d\n' % args.train_loader_worker)
        logfp.write('test_loader_worker         = %d\n' % args.test_loader_worker)
        logfp.write('use_multiprocessing        = %s\n' % args.use_multiprocessing)
        logfp.write('select_by_aupr             = %s\n' % args.select_by_aupr)
        logfp.write('select_by_corr             = %s\n' % args.select_by_corr)
        logfp.write('balance_method             = %d\n' % balance_method)
        logfp.write('weighting_method           = %d\n' % weighting_method)
        logfp.write('train_partial              = %0.2f\n' % args.train_partial)
        logfp.write('class_weight               = [%s]\n' % ', '.join(['%0.4f' % x for x in class_weight]))
        logfp.write('alpha                      = [%s]\n' % ', '.join(['%0.1f' % (x * 100) for x in args.alpha]))
        logfp.write('alpha_topk                 = [%s]\n' % ', '.join(['%d' % x for x in args.alpha_topk]))
        logfp.write('score levels               = [%s]\n' % ', '.join(['%0.2f' % x for x in args.score_levels]))
        logfp.write('early_stop                 = %d\n' % args.early_stop)
        logfp.write('use_pretrained             = %d\n' % use_pretrained)
        if args.embedding_file is not None:
            logfp.write('embedding_file             = %s\n' % args.embedding_file)
        logfp.write('\n#----- Training Log ------#\n')
        logfp.flush()

    # --- (6) train ---#
    queue_size = 3
    if args.use_multiprocessing:
        data_container = multiprocessing.Queue
        # data_container   = multiprocessing.Manager().Queue
    else:
        data_container = queue.Queue
    train_data_queue = data_container(queue_size * args.train_loader_worker)
    test_data_queue = data_container(queue_size * args.test_loader_worker)

    test_DLM = data_loader.Data_Loader_Manager(batch_data_loader=model_agent.batch_data_loader,
                                               data_queue=test_data_queue,
                                               data=(mol_features_test, ground_truths_test),
                                               shuffle=False,
                                               batch_size=batch_size_test,
                                               worker_num=args.test_loader_worker,
                                               use_multiprocessing=args.use_multiprocessing,
                                               auto_rewind=0,
                                               name='test_DLM')

    vprint('Start training', l=1)
    ER_history = finite_memory_array(np.ones(shape=(2, accum_loss_num)) * 1.0)  # classification error rate history
    best_ER_train, best_ER_test = 100.0, 100.0
    best_pearson_train, best_pearson_test = -1.0, -1.0  # todo: @jinjiang
    best_spearman_train, best_spearman_test = -1.0, -1.0  # todo: @jinjiang
    best_rmse_train, best_rmse_test = 1000.0, 1000.0
    no_better_test_times = 0

    best_aupr = 0.0
    best_aupr_values = [0.0, 0.0]
    best_aupr_positive = [0.0, 0.0]
    best_corr = 0.0
    total_trained_sample_num = 0
    total_trained_batch_num = 0
    # if use_pretrained > 0:
    #     lr_lambda    = lambda epoch: 1.0 if epoch <= use_pretrained else 0.2
    #     lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    for epoch in range(epoch_start, args.max_epoch_num):
        if no_better_test_times > args.early_stop > 0:
            break
        trainlog.epoch = epoch
        epoch_time0 = time.time()
        total_sample, total_wrong, ER_batch, ER_train = 0.0, 0.0, 100.0, 100.0
        trained_sample_num, tmp_model_saved_num, tmp_model_tested_num = 0, 0, 0
        # ---- (6.1) balance resamping ----#
        # 0: no balance
        if balance_method >1 and args.class_num <=1:
            warnings.warn('regression task DOES NOT support data resampling, No resample is adapted')
            balance_method = 0
        if balance_method == 1:  # sqrt of original distribution
            vprint('balance sampling, method 1')
            class_distribution = calc_class_distribution(ground_truths_train, class_num=args.class_num)
            expected_class_distribution = np.sqrt(class_distribution)
            mol_features_train_epoch, ground_truths_train_epoch, SMILES_train_epoch, IDs_train_epoch, sample_weights_train_epoch = \
                balance_resample(
                    (mol_features_train, ground_truths_train, SMILES_train, IDs_train, sample_weights_train),
                    ground_truths_train,
                    class_num=args.class_num,
                    expected_class_distribution=expected_class_distribution,
                    expected_sample_num=None,
                    shuffled=True)
        elif balance_method == 2:  # uniform distribution
            vprint('balance sampling, method 2')
            expected_class_distribution = np.ones(args.class_num) / args.class_num
            mol_features_train_epoch, ground_truths_train_epoch, SMILES_train_epoch, IDs_train_epoch, sample_weights_train_epoch = \
                balance_resample(
                    (mol_features_train, ground_truths_train, SMILES_train, IDs_train, sample_weights_train),
                    ground_truths_train,
                    class_num=args.class_num,
                    expected_class_distribution=expected_class_distribution,
                    expected_sample_num=None,
                    shuffled=True)
        else:
            mol_features_train_epoch, ground_truths_train_epoch, SMILES_train_epoch, IDs_train_epoch, sample_weights_train_epoch = \
                mol_features_train, ground_truths_train, SMILES_train, IDs_train, sample_weights_train
            vprint('No balance sampling')
            pass

        train_sample_num = len(mol_features_train_epoch)
        save_model_per_sample_num = int(train_sample_num * args.save_model_per_epoch_num)
        test_model_per_sample_num = int(train_sample_num * args.test_model_per_epoch_num)
        vprint('save_model_per_sample_num = ', save_model_per_sample_num, l=1)
        vprint('test_model_per_sample_num = ', test_model_per_sample_num, l=1)

        train_DLM = data_loader.Data_Loader_Manager(batch_data_loader=model_agent.batch_data_loader,
                                                    data_queue=train_data_queue,
                                                    data=(mol_features_train_epoch, ground_truths_train_epoch,
                                                          sample_weights_train_epoch),
                                                    shuffle=True,
                                                    batch_size=batch_size,
                                                    batch_size_min=batch_size_min,
                                                    worker_num=args.train_loader_worker,
                                                    use_multiprocessing=args.use_multiprocessing,
                                                    auto_rewind=0,
                                                    name='train_DLM')
        train_sample_num = ground_truths_train_epoch.shape[0]
        while trained_sample_num < train_sample_num:
            # while trained_sample_num < 4:
            if no_better_test_times > args.early_stop > 0:
                break
            time0 = time.time()
            model_agent.model.train(True)
            if use_pretrained > 0:
                if args.model_ver == '4v4':
                    if epoch < use_pretrained:
                        for module in [model_agent.model.emb0, model_agent.model.blocks, model_agent.model.readout_ops]:
                            freeze_module(module)
                        # vprint('CL pretrained modules freezed', l=1)
                    else:
                        for module in [model_agent.model.emb0, model_agent.model.blocks, model_agent.model.readout_ops]:
                            unfreeze_module(module)
                        # vprint('CL pretrained modules unfreezed', l=1)
                elif args.model_ver == '4v6':
                    if epoch < use_pretrained:
                        for module in [model_agent.model.emb0, model_agent.model.blocks]:
                            freeze_module(module)
                        # vprint('4v6 pretrained modules freezed', l=1)
                    else:
                        for module in [model_agent.model.emb0, model_agent.model.blocks]:
                            unfreeze_module(module)
                        # vprint('4v6 pretrained modules unfreezed', l=1)

            # ---- train step ----#
            batch_train_data = train_data_queue.get()
            time1 = time.time()
            # model_agent.model.train(False)
            try:
                X, Y, sample_weight = batch_train_data[0], batch_train_data[-2], batch_train_data[-1]
                if hasattr(args, 'anneal_dropouts'):
                    dropout = get_annealing_dropout(epoch, args.anneal_dropouts, args.max_epoch_num)
                else:
                    dropout = 0
                scorematrix, loss = model_agent.forward(batch_train_data, calc_loss=True, dropout=dropout,
                                                        class_weight=class_weight,
                                                        sample_weight=sample_weight)
                if torch.all(torch.isnan(loss)):
                    raise ValueError('loss is all nan!')
                if accum_loss_num > 0:
                    loss = loss/accum_loss_num
                loss.backward()
                total_trained_batch_num += 1
                if accum_loss_num > 0 and total_trained_batch_num % accum_loss_num == 0:
                    print('gradients updated @ %d every %d' % (total_trained_batch_num, accum_loss_num))
                    optimizer.step()
                    optimizer.zero_grad()
                elif accum_loss_num == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                # if use_pretrained > 0:
                #     lr_scheduler.step()
            except MemoryError:
                warnings.warn('Memory error encountered')
                vprint('total_nodes in batch = %d' % X.shape[0], l=10)
                clear_cuda_cache(model_agent.model)
                batch_sample_num = Y.shape[0]
                trained_sample_num += batch_sample_num
                continue
            except RuntimeError as e:
                vprint('RuntimeError catched', l=10)
                if e.args[0].startswith('CUDA out of memory'):
                    warnings.warn('CUDA out of memory encountered')
                    vprint('total_nodes in batch = %d' % X.shape[0], l=10)
                    clear_cuda_cache(model_agent.model)
                    batch_sample_num = Y.shape[0]
                    trained_sample_num += batch_sample_num
                    continue
                else:
                    raise e
            # ---- metric calculation ----#
            if args.task == 'regression':  # for regression problem
                RMSE_train = np.sqrt(((scorematrix.squeeze(1).cpu().detach().numpy() - Y) ** 2).mean())
                pearson_corr, pp_value = metrics.pearson_corr(scorematrix.cpu().squeeze(1).detach().numpy(), Y)
                spearman_corr, sp_value = metrics.spearman_corr(scorematrix.cpu().squeeze(1).detach().numpy(), Y)
                std_AE_train = np.std(np.abs(scorematrix.squeeze(1).cpu().detach().numpy() - Y))

                batch_sample_num = Y.shape[0]
                trained_sample_num += batch_sample_num
                total_trained_sample_num += batch_sample_num
                time2 = time.time()
                data_process_time = time1 - time0
                train_time = time2 - time1
                progress = trained_sample_num / train_sample_num
                loss = loss.cpu().detach().numpy()
                vprint(
                    'ep %d, batch %d, loss=%0.6f, RMSE_train: %0.4f, std_AE_train: %0.4f, Pearson: %0.2f, Spearman: %0.2f, time = %0.2fs(%0.2f|%0.2f), progress = %0.2f%%, time remained = %0.2fh' % (
                        epoch, total_trained_batch_num, loss, RMSE_train, std_AE_train, pearson_corr, spearman_corr,
                        (time2 - time0), train_time, data_process_time,
                        progress * 100.0, (time.time() - epoch_time0) / 3600 * (1 - progress) / max(progress, 1e-6)))
                # ---- save temporary model ----#
                if int(trained_sample_num / save_model_per_sample_num) > tmp_model_saved_num:
                    tmp_model_saved_num += 1
                    logfp.write('%s   RMSE_train: %0.4f, Pearson: %0.2f, Spearman: %0.2f, trained_sample_num = %d\n' % (
                    get_time_stamp(), RMSE_train, pearson_corr, spearman_corr, trained_sample_num))
                    logfp.flush()
                    if RMSE_train < best_rmse_train:
                        best_rmse_train = RMSE_train
                        tmp_model_file = os.path.join(model_folder,
                                                      'tmp_[%s]_ligand_based_VS_regression_model_%s@%s_N=%d_RSME=%0.2f.pt' %
                                                      (prefix, args.model_ver, time_stamp, total_trained_sample_num,
                                                       pearson_corr))
                        model_agent.dump_weight(tmp_model_file)
                        vprint('temporary model data saved')
                        trainlog.best_tmp_model = tmp_model_file
                        trainlog.best_rmse_train = best_rmse_train

            else:  # for classification
                scorematrix = softmax(scorematrix, dim=1)
                score_mask_neg = torch.unsqueeze((scorematrix[:, 0] >= args.pred_threshold), 1)  #threshold negetive
                score_mask_pos = torch.unsqueeze((scorematrix[:, 1] > (1 - args.pred_threshold)), 1)  #threshold positive
                score_mask = torch.cat((score_mask_neg, score_mask_pos), dim =1)
                best_labels = torch.argmax(score_mask.long(), dim=1)
                best_ps = torch.masked_select(scorematrix, score_mask)
                # best_ps, best_labels = torch.max(scorematrix, dim=1)
                prediction = best_labels.cpu().detach().numpy().astype(np.int32)
                wrong_sample_num = np.sum(prediction != Y)
                batch_sample_num = Y.shape[0]
                ER_history.update(np.array([wrong_sample_num, batch_sample_num]))
                total_wrong = ER_history.content[0].sum()
                total_sample = ER_history.content[1].sum()
                ER_train = total_wrong * 100.0 / total_sample
                ER_train_batch = wrong_sample_num / batch_sample_num * 100.0
                trained_sample_num += batch_sample_num
                total_trained_sample_num += batch_sample_num
                time2 = time.time()
                data_process_time = time1 - time0
                train_time = time2 - time1
                progress = trained_sample_num / train_sample_num
                loss = loss.cpu().detach().numpy()
                vprint(
                    'ep %d, batch %d, loss=%0.6f, ER = %0.2f|%0.2f, time = %0.2fs(%0.2f|%0.2f), progress = %0.2f%%, time remained = %0.2fh' % (
                        epoch, total_trained_batch_num, loss, ER_train, ER_train_batch, (time2 - time0), train_time,
                        data_process_time,
                        progress * 100.0, (time.time() - epoch_time0) / 3600 * (1 - progress) / max(progress, 1e-6)))

                # ---- save temporary model ----#
                if int(trained_sample_num / save_model_per_sample_num) > tmp_model_saved_num:
                    tmp_model_saved_num += 1
                    logfp.write('%s   ER_train = %0.2f, trained_sample_num = %d\n' % (
                    get_time_stamp(), ER_train, trained_sample_num))
                    logfp.flush()
                    if ER_train < best_ER_train:
                        best_ER_train = ER_train
                        tmp_model_file = os.path.join(model_folder,
                                                      'tmp_[%s]_ligand_based_VS_model_%s@%s_N=%d_ER=%0.2f.pt' %
                                                      (prefix, args.model_ver, time_stamp, total_trained_sample_num,
                                                       ER_train))
                        model_agent.dump_weight(tmp_model_file)
                        vprint('temporary model data saved')
                        trainlog.best_tmp_model = tmp_model_file
                        trainlog.best_ER_train = best_ER_train

            # ---- evaluate model on test set ----#
            if args.task == 'regression':  # for regression
                start_test_sign = RMSE_train - args.RMSE_start_test
            else:  # for classification
                start_test_sign = ER_train - args.ER_start_test

            if int(trained_sample_num / test_model_per_sample_num) > tmp_model_tested_num and start_test_sign <= 0:
                if args.task == 'regression':  # for regression
                    vprint('RMSE_train = %0.4f' % RMSE_train, l=1)
                else:  # for classification
                    vprint('ER_train = %0.2f' % ER_train, l=1)

                tmp_model_tested_num += 1
                vprint('evaluating model on test set (size = %d)' % test_sample_num, l=1)
                test_time0 = time.time()
                tested_sample_num, total_wrong, ER_test = 0, 0, 100.0
                model_agent.model.train(False)
                groundtruths = []
                scorematrix_all = []
                # graphs_all = []
                with torch.no_grad():
                    while tested_sample_num < test_sample_num:
                        batch_test_data = test_data_queue.get()
                        # test_time1 = time.time()
                        try:
                            X, Y = batch_test_data[0], batch_test_data[-1]
                            scorematrix, *graphs = model_agent.forward(batch_test_data)
                            # graphs_all.append(graphs[0].detach().cpu().numpy())
                        except MemoryError:
                            warnings.warn('Memory error encountered')
                            vprint('total_nodes in batch = %d' % X.shape[0], l=10)
                            clear_cuda_cache(model_agent.model)
                            continue
                        except RuntimeError as e:
                            vprint('RuntimeError catched', l=10)
                            if e.args[0].startswith('cuda runtime error (2) : out of memory'):
                                warnings.warn('Memory error(2) encountered')
                                vprint('total_nodes in batch = %d' % X.shape[0], l=10)
                                clear_cuda_cache(model_agent.model)
                                continue
                            else:
                                raise e
                        # ---- metric calculation ----#
                        if args.task == 'regression':
                            scorematrix_all.append(scorematrix.cpu().detach().numpy())
                            RMSE_test = np.sqrt(((scorematrix.squeeze(1).cpu().detach().numpy() - Y) ** 2).mean())
                            std_AE_test = np.std(np.abs(scorematrix.squeeze(1).cpu().detach().numpy() - Y))
                            # pearson_corr_test, pp_value_test = sp.stats.pearsonr(scorematrix.squeeze(1).detach().numpy(), Y)
                            # spearman_corr_test, sp_value_test = sp.stats.spearmanr(scorematrix.squeeze(1).detach().numpy(), Y)

                            batch_sample_num = Y.shape[0]
                            tested_sample_num += batch_sample_num
                            groundtruths.append(Y)

                            test_time1 = time.time()
                            progress = tested_sample_num / test_sample_num
                            vprint(
                                'RMSE_test = %0.4f, std_AE_test = %0.4f, progress = %0.2f%%, time remained = %0.2fmins' % (
                                    RMSE_test, std_AE_test, progress * 100,
                                    (test_time1 - test_time0) / 60 * (1 - progress) / progress))
                        else:
                            scorematrix = softmax(scorematrix, dim=1)
                            scorematrix_all.append(scorematrix.cpu().detach().numpy())
                            score_mask_neg = torch.unsqueeze((scorematrix[:, 0] >= args.pred_threshold), 1)  #set threshold negitve
                            score_mask_pos = torch.unsqueeze((scorematrix[:, 1] > (1 - args.pred_threshold)), 1)  #set threshold postive
                            score_mask = torch.cat((score_mask_neg, score_mask_pos), dim =1)
                            best_labels = torch.argmax(score_mask.long(), dim=1)
                            best_ps = torch.masked_select(scorematrix, score_mask)
                            # best_ps, best_labels = torch.max(scorematrix, dim=1)
                            prediction = best_labels.cpu().detach().numpy().astype(np.int32)
                            wrong_sample_num = np.sum(prediction != Y)
                            batch_sample_num = Y.shape[0]
                            tested_sample_num += batch_sample_num
                            groundtruths.append(Y)

                            total_wrong += wrong_sample_num
                            ER_test = total_wrong / tested_sample_num * 100.0
                            test_time1 = time.time()
                            progress = tested_sample_num / test_sample_num
                            vprint('ER_test = %0.2f, progress = %0.2f%%, time remained = %0.2fmins' % (
                                ER_test, progress * 100, (test_time1 - test_time0) / 60 * (1 - progress) / progress))
                test_time2 = time.time()
                groundtruths = np.concatenate(groundtruths)
                scorematrix = np.concatenate(scorematrix_all, axis=0)
                no_better_test_times += 1
                test_DLM.rewind()

                if args.task == 'regression':  # for regression
                    pearson_corr_all, pp_value_all = metrics.pearson_corr(scorematrix.squeeze(1), groundtruths)
                    spearman_corr_all, sp_value_all = metrics.spearman_corr(scorematrix.squeeze(1), groundtruths)
                    if pearson_corr_all <= 0.0 or spearman_corr_all <= 0.0:
                        corr_hmean = 0.0
                    else:
                        corr_hmean = sp.stats.hmean([pearson_corr_all, spearman_corr_all])
                    if corr_hmean > best_corr:
                        best_corr = corr_hmean
                        if args.select_by_corr:
                            best_model_file = os.path.join(model_folder,
                                                           '[%s]_ligand_based_VS_model_%s@%s_N=%d_corrhmean=%0.2f_RMSE_stdAE=[%0.2f, %0.2f].pt' %
                                                           (
                                                           prefix, args.model_ver, time_stamp, total_trained_sample_num,
                                                           best_corr, RMSE_test, std_AE_test))
                            model_agent.dump_weight(best_model_file)
                            trainlog.best_model = best_model_file
                            trainlog.best_corr = best_corr
                            no_better_test_times = 0
                    if RMSE_test < best_rmse_test:
                        best_rmse_test = RMSE_test
                        if not args.select_by_corr:
                            best_model_file = os.path.join(model_folder,
                                                           '[%s]_ligand_based_VS_model_%s@%s_N=%d_RMSE_stdAE=[%0.4f, %0.4f]_corrhmean=%0.2f.pt' %
                                                           (
                                                           prefix, args.model_ver, time_stamp, total_trained_sample_num,
                                                           RMSE_test, std_AE_test, best_corr))
                            model_agent.dump_weight(best_model_file)
                            trainlog.best_model = best_model_file
                            trainlog.best_rmse_test = best_rmse_test
                            no_better_test_times = 0
                    test_time_cost = test_time2 - test_time0
                    logfp.write(
                        '%s   RMSE_test = %0.4f, RMSE_train = %0.4f, trained_sample_num = %d, test time cost = %0.2f mins, speed = %0.2f samples/s\n' % (
                            get_time_stamp(), RMSE_test, RMSE_train, trained_sample_num, test_time_cost / 60,
                            tested_sample_num / test_time_cost))
                    logfp.write(
                        '%s   std_AE_test = %0.4f, std_AE_train = %0.4f, trained_sample_num = %d, test time cost = %0.2f mins, speed = %0.2f samples/s\n' % (
                            get_time_stamp(), std_AE_test, std_AE_train, trained_sample_num, test_time_cost / 60,
                            tested_sample_num / test_time_cost))
                    logfp.write('Pearson_Spearman = [%0.4f, %0.4f] \n' % (pearson_corr_all, spearman_corr_all))
                    logfp.flush()
                    vprint(
                        'RMSE_test = %0.4f, std_AE_test = %0.4f, epoch = %d, time cost = %0.2fs, speed = %0.2f samples/s' %
                        (RMSE_test, std_AE_test, epoch, test_time_cost, tested_sample_num / test_time_cost), l=1)
                    vprint('Pearson_Spearman = [%0.4f, %0.4f] \n' % (pearson_corr_all, spearman_corr_all), l=1)
                else:  # for classification
                    roc_curves = metrics.roc_curve(groundtruths, scorematrix)
                    pr_curves = metrics.pr_curve(groundtruths, scorematrix)

                    aurocs = [x[3] for x in roc_curves]
                    auprs = [x[3] for x in pr_curves]
                    try:
                        aupr_hmean = sp.stats.hmean(auprs)
                    except:
                        aupr_hmean = 0.0
                    if auprs[1] > best_aupr_positive[1]:
                        best_aupr_positive = auprs[:]
                    if aupr_hmean >= best_aupr:
                        best_aupr = aupr_hmean
                        best_aupr_values = auprs
                        if args.select_by_aupr:
                            best_model_file = os.path.join(model_folder,
                                                           '[%s]_ligand_based_VS_model_%s@%s_N=%d_aupr=%0.2f_ER=[%0.2f, %0.2f].pt' %
                                                           (
                                                           prefix, args.model_ver, time_stamp, total_trained_sample_num,
                                                           best_aupr, ER_test, ER_train))
                            model_agent.dump_weight(best_model_file)
                            trainlog.best_model = best_model_file
                            trainlog.best_aupr = best_aupr
                            no_better_test_times = 0
                    if ER_test < best_ER_test:
                        best_ER_test = ER_test
                        if not args.select_by_aupr:
                            best_model_file = os.path.join(model_folder,
                                                           '[%s]_ligand_based_VS_model_%s@%s_N=%d_ER=[%0.2f, %0.2f]_aupr=%0.2f.pt' %
                                                           (
                                                           prefix, args.model_ver, time_stamp, total_trained_sample_num,
                                                           ER_test, ER_train, best_aupr))
                            model_agent.dump_weight(best_model_file)
                            trainlog.best_model = best_model_file
                            trainlog.best_ER_test = best_ER_test
                            no_better_test_times = 0
                    test_time_cost = test_time2 - test_time0
                    logfp.write(
                        '%s   ER_test = %0.2f, ER_train = %0.2f, trained_sample_num = %d, test time cost = %0.2f mins, speed = %0.2f samples/s\n' % (
                            get_time_stamp(), ER_test, ER_train, trained_sample_num, test_time_cost / 60,
                            tested_sample_num / test_time_cost))
                    logfp.write('AuROCs = [%s], AuPRs = [%s] \n' % (
                    ', '.join(['%0.2f' % x for x in aurocs]), ', '.join(['%0.2f' % x for x in auprs])))
                    logfp.flush()
                    vprint('ER_test=%0.2f, epoch = %d, time cost = %0.2fs, speed = %0.2f samples/s' %
                           (ER_test, epoch, test_time_cost, tested_sample_num / test_time_cost), l=1)
                    vprint('AuROCs = [%s], AuPRs = [%s]' % (
                    ', '.join(['%0.2f' % x for x in aurocs]), ', '.join(['%0.2f' % x for x in auprs])), l=1)

                    class_sample_num = []
                    class_precision = []
                    class_recall = []
                    class_F1 = []
                    # prediction = np.argmax(scorematrix, axis=1)
                    score_mask_neg_all = np.expand_dims((scorematrix[:, 0] >= args.pred_threshold), axis=1)  #set threshold negitve
                    score_mask_pos_all = np.expand_dims((scorematrix[:, 1] > (1 - args.pred_threshold)), axis=1)  #set threshold postive
                    score_mask = np.concatenate((score_mask_neg_all, score_mask_pos_all), axis=1)
                    prediction = np.argmax(score_mask, axis=1).astype(np.int64)
                    confusion_matrix, _ = compute_confusion_matrix(prediction, groundtruths, args.class_num)
                    for i in range(args.class_num):
                        precision = confusion_matrix[i, i] / confusion_matrix[:, i].sum() * 100.0
                        recall = confusion_matrix[i, i] / confusion_matrix[i, :].sum() * 100.0
                        if precision + recall > 0.0:
                            F1 = 2 * precision * recall / (precision + recall)
                        else:
                            F1 = 0.0
                        class_sample_num.append(confusion_matrix[i, :].sum())
                        class_precision.append(precision)
                        class_recall.append(recall)
                        class_F1.append(F1)
                        vprint('class_%d: P = %0.2f, R = %0.2f, F1 = %0.2f' % (i, precision, recall, F1), l=1)
                        logfp.write('class_%d: P = %0.2f, R = %0.2f, F1 = %0.2f\n' % (i, precision, recall, F1))

                    class_sample_num = np.array(class_sample_num)
                    class_precision = np.array(class_precision)
                    class_recall = np.array(class_recall)
                    class_F1 = np.array(class_F1)
                    each_class_weight = class_sample_num / groundtruths.size
                    weighted_avgP = (each_class_weight * class_precision).sum()
                    weighted_avgR = (each_class_weight * class_recall).sum()
                    weighted_avgF1 = (each_class_weight * class_F1).sum()
                    if np.all(class_precision > 0.0):
                        macro_avgP = sp.stats.hmean(class_precision)
                    else:
                        macro_avgP = 0.0
                    if np.all(class_recall > 0.0):
                        macro_avgR = sp.stats.hmean(class_recall)
                    else:
                        macro_avgR = 0.0
                    if np.all(class_F1 > 0.0):
                        macro_avgF1 = sp.stats.hmean(class_F1)
                    else:
                        macro_avgF1 = 0.0

                    vprint('weighted_avg_P = %0.2f, weighted_avg_R = %0.2f, weighted_avg_F1 = %0.2f' % (
                    weighted_avgP, weighted_avgR, weighted_avgF1), l=1)
                    logfp.write('weighted_avg_P = %0.2f, weighted_avg_R = %0.2f, weighted_avg_F1 = %0.2f\n' % (
                    weighted_avgP, weighted_avgR, weighted_avgF1))
                    vprint('macro_avg_P = %0.2f, macro_avg_R = %0.2f, macro_avg_F1 = %0.2f' % (
                    weighted_avgP, weighted_avgR, weighted_avgF1), l=1)
                    logfp.write('macro_avg_P = %0.2f, macro_avg_R = %0.2f, macro_avg_F1 = %0.2f\n' % (
                    macro_avgP, macro_avgR, macro_avgF1))

                    if args.class_num == 2:
                        alpha_acc_list, relative_enrichment_list, positive_ratio = compute_alpha_enrichment(
                            groundtruths, scorematrix[:, 1], args.alpha)
                        for a, alpha_acc, relative_enrichment in zip(args.alpha, alpha_acc_list,
                                                                     relative_enrichment_list):
                            logfp.write('Alpha_enrichment: (%0.2f, %0.2f, %0.2f)@%0.1f\n' % (
                            alpha_acc * 100, positive_ratio * 100, relative_enrichment, a * 100))
                            vprint('Alpha_enrichment: (%0.2f, %0.2f, %0.2f)@%0.1f\n' % (
                            alpha_acc * 100, positive_ratio * 100, relative_enrichment, a * 100), l=1)
                        n = groundtruths.size
                        alpha2 = np.array(args.alpha_topk) / n
                        alpha_acc_list2, relative_enrichment_list2, positive_ratio = compute_alpha_enrichment(
                            groundtruths, scorematrix[:, 1], alpha2)
                        for a, alpha_acc, relative_enrichment in zip(args.alpha_topk, alpha_acc_list2,
                                                                     relative_enrichment_list2):
                            logfp.write('Topk_enrichment: (%0.2f, %0.2f, %0.2f)@%d\n' % (
                            alpha_acc * 100, positive_ratio * 100, relative_enrichment, a))
                            vprint('Topk_enrichment: (%0.2f, %0.2f, %0.2f)@%d\n' % (
                            alpha_acc * 100, positive_ratio * 100, relative_enrichment, a), l=1)
                        level_acc_list, relative_enrichment_list, positive_ratio = compute_score_level_enrichment(
                            groundtruths, scorematrix[:, 1], args.score_levels)
                        for a, level_acc, relative_enrichment in zip(args.score_levels, level_acc_list,
                                                                     relative_enrichment_list):
                            logfp.write('Score_level_enrichment: (%0.2f, %0.2f, %0.2f)@%0.2f\n' % (
                            level_acc * 100, positive_ratio * 100, relative_enrichment, a))
                            vprint('Score_level_enrichment: (%0.2f, %0.2f, %0.2f)@%0.2f\n' % (
                            level_acc * 100, positive_ratio * 100, relative_enrichment, a), l=1)
                    print('\n')
        train_DLM.close()
        epoch_time1 = time.time()
        if args.class_num == 1:
            logfp.write('epoch = %d done, RMSE_train = %0.4f, std_AE_train = %0.4f, time = %0.2f mins\n' % (
            epoch, RMSE_train, std_AE_train, (epoch_time1 - epoch_time0) / 60))
            logfp.flush()
        else:
            logfp.write('epoch = %d done, ER_train = %0.2f, time = %0.2f mins\n' % (
            epoch, ER_train, (epoch_time1 - epoch_time0) / 60))
            logfp.flush()

    logfp.write('All done~\n')
    logfp.close()
    test_DLM.close()
    for p in multiprocessing.active_children():
        p.terminate()

    # --- plot convergence curves ---#
    if args.task == 'regression':
        plot_regress_convergence_curve(log_file=logfile, save_fig=True, noshow=True)
    else:
        plot_convergence_curve(log_file=logfile, save_fig=True, noshow=True, summary=True)
    vprint('============== Train Log ====================', l=1)
    for key in trainlog.__dict__:
        print(key, '=', trainlog.__dict__[key])
    vprint('=============================================\n', l=1)
    vprint('All done~', l=10)
    print('Auprs at the best hmean aupr', best_aupr,  best_aupr_values)
    print('Best positive aupr', best_aupr_positive)
    # --- scaffold analysis ---#
    vprint('============== Scaffold Analysis ====================', l=1)
    idxs_trainset, scaffold_dict_trainset = get_scaffold_group(SMILES_train, include_chirality=True)
    idxs_testset, scaffold_dict_testset = get_scaffold_group(SMILES_test, include_chirality=True)
    vprint('Trainset: %s' % scaffold_summary(scaffold_dict_trainset), l=1)
    vprint('Testset : %s' % scaffold_summary(scaffold_dict_testset), l=1)
    summary = scaffold_overlap_summary(scaffold_dict_trainset, scaffold_dict_testset)[0]
    for s in summary:
        vprint(s, l=1)

    # --- clean ---#
    sys.stdout = stdout_tap.stream
    sys.stderr = stderr_tap.stream
    stdout_tap_log.close()
    stderr_tap_log.close()
    return trainlog


# --- multi-fold cross validation training ---#
def multi_fold_train(args):

    if args.grid_search > 0:
        raise ValueError('-mfold and -grid_search should not be enabled simultaneously')
    if args.testset is not None:
        raise ValueError('For multi-fold cross validation, input testset %s should be None' % args.testset)

    mfold_indexs = None
    dataset_all = None
    num_local_sample = args.num_local_sample
    training_strategy = args.training_strategy

    if args.srcfolder is not None:
        if args.trainset is not None:
            raise ValueError('MFOLD: srcfolder and trainset should not be specified simultaneously')
        if os.path.exists(args.srcfolder):
            # ---- load mfold index ----#
            existing_mfold_index_files = [x.name for x in os.scandir(args.srcfolder) if
                                          x.is_file() and x.name.startswith('mfold_indexs')]
            if len(existing_mfold_index_files) != 1:
                raise ValueError(
                    'MFOLD: %d mfold index files located under %s' % (len(existing_mfold_index_files), args.srcfolder))
            mfold_index_file = os.path.join(args.srcfolder, existing_mfold_index_files[0])
            mfold_indexs = gpickle.load(mfold_index_file)
            if len(mfold_indexs) != int(args.mfold):
                raise ValueError('MFOLD: args.mfold = %s, whereas %d folds index loaed from %s' % (
                args.mfold, len(mfold_indexs), mfold_index_file))
            args.mfold = int(args.mfold)
            # ----- load trainset ----#
            existing_trainset_files = [x.name for x in os.scandir(args.srcfolder) if
                                       x.is_file() and x.name.startswith('trainset')]
            if len(existing_trainset_files) != 1:
                raise ValueError(
                    'MFOLD: %d trainset files located under %s' % (len(existing_trainset_files), args.srcfolder))
            trainset_file = os.path.join(args.srcfolder, existing_trainset_files[0])
            dataset_all = gpickle.load(trainset_file)
            args.srcfolder = None
        else:
            raise ValueError('MFOLD: srcfolder not exist: %s' % args.srcfolder)
    else:
        if os.path.exists(args.mfold):  # mfold_indexs.gpkl exists
            mfold_indexs = gpickle.load(args.mfold)
            print('mfold_indexs loaded from %s' % args.mfold)
            args.mfold = len(mfold_indexs)
        else:
            args.mfold = int(args.mfold)

        if args.save_root_folder is not None:
            trainset_file = os.path.join(args.save_root_folder, 'trainset.gpkl')
            mfold_indexes_file = os.path.join(args.save_root_folder, 'mfold_indexs.gpkl')
            if os.path.exists(trainset_file):
                if args.trainset is not None:
                    warnings.warn(
                        'args.trainset will be ignored when "trainset.gpkl" is found under `save_root_folder`')
            else:
                trainset_file = args.trainset
            dataset_all = gpickle.load(trainset_file)

            if os.path.exists(mfold_indexes_file):
                mfold_indexs = gpickle.load(mfold_indexes_file)
                assert args.mfold == len(mfold_indexs)
                print('`mfold_indexs.gpkl` located & loaded')
        else:
            dataset_all = gpickle.load(args.trainset)

    if isinstance(dataset_all, dict):
        mol_features_all = dataset_all['features']
        ground_truths_all = dataset_all['labels']
        SMILES_all = dataset_all['SMILESs']
        IDs_all = dataset_all['IDs']
        sample_weights_all = dataset_all['sample_weights']
    else:  # for back-compatibility
        mol_features_all, ground_truths_all, SMILES_all, IDs_all, *sample_weights_all = dataset_all
        sample_weights_all = sample_weights_all[0] if len(sample_weights_all) > 0 else None
    # --- forced type cast ---#
    if args.task == 'regression':
        ground_truths_all = ground_truths_all.astype(np.float32)  # regression:
    else:
        ground_truths_all = ground_truths_all.astype(np.int64)  # classification

    # --- m-fold index ---#
    if mfold_indexs is None:
        mfold_indexs = [[] for _ in range(args.mfold)]
        sample_num_all = SMILES_all.shape[0]
        if args.group_by_scaffold:
            if not isinstance(dataset_all, dict):
                raise ValueError('dataset is required to be dict typed when `group_by_scaffold` is enabled')
            scaffold_dict = dataset_all['scaffold_dict']
            scaffold_dict_keys = list(scaffold_dict)
            np.random.shuffle(scaffold_dict_keys)
            index = []
            for key in scaffold_dict_keys:
                np.random.shuffle(scaffold_dict[key])
                index.extend(scaffold_dict[key])
        else:
            index = np.arange(sample_num_all)

            if training_strategy:
                # Update 2021/8/6: Add new CV strategy
                index_global = index[:-num_local_sample]
                index_cv = index[-num_local_sample:]
                index = index_cv

            np.random.shuffle(index)

        # Update: 2021/8/15 A patch of `split into m part per class`
        if training_strategy:
            # start = len(index_global)
            # stop = start + len(index)
            mfold_indexs = split_y(
                y=[ground_truths_all[idx] for idx in index],
                indices=index,
                n_splits=args.mfold,
                stratify=True,
            )
        else:
            # ---- split into m part per class ----#
            for label in range(args.class_num):
                if training_strategy:
                    sample_idxs_perclass = [index[i] for i in range(len(index)) if
                                            ground_truths_all[
                                                index[i]] == label]  # get ids of data with current class label
                else:
                    sample_idxs_perclass = [index[i] for i in range(sample_num_all) if
                                            ground_truths_all[index[i]] == label]  # get ids of data with current class label
                step = len(sample_idxs_perclass) // args.mfold
                start = 0
                for i in range(args.mfold):
                    if i < args.mfold - 1:
                        end = start + step
                    else:
                        end = len(sample_idxs_perclass)
                    mfold_indexs[i].extend(sample_idxs_perclass[start:end])
                    start = end

    # --- other initialization ---#
    if args.save_root_folder is None:
        if args.prefix is not None:
            args.save_root_folder = os.path.join(os.getcwd(), 'train_results/[%s]_%d_fold_%s_model_%s@%s' % (
            args.prefix, args.mfold, args.task, args.model_ver, get_time_stamp()))
        else:
            args.save_root_folder = os.path.join(os.getcwd(), 'train_results/%d_fold_%s_model_%s@%s' % (
            args.mfold, args.task, args.model_ver, get_time_stamp()))
    if not os.path.exists(args.save_root_folder):
        os.makedirs(args.save_root_folder)

    # --- run training ---#
    for i in range(args.mfold):
        # ---- scan for existing fold folders ----#
        existing_mfold_folders = [x.name for x in os.scandir(args.save_root_folder) if
                                  x.is_dir() and x.name.startswith('[Fold')]
        existing_mfold_indexs = set()
        for item in existing_mfold_folders:
            endidx = item.find(']')
            idx = int(item[5:endidx].split(',')[0].split('-')[0])
            existing_mfold_indexs.add(idx)
        if i in existing_mfold_indexs:
            continue

        # ---- train a fold ----#
        if i == 0:
            trainset_file = os.path.join(args.save_root_folder, 'trainset.gpkl')
            mfold_indexes_file = os.path.join(args.save_root_folder, 'mfold_indexs.gpkl')
            if not os.path.exists(mfold_indexes_file):
                gpickle.dump(mfold_indexs, mfold_indexes_file)
            if not os.path.exists(trainset_file):
                gpickle.dump(dataset_all, trainset_file)
        fold_args = copy.deepcopy(args)
        index_for_testset = mfold_indexs[i]
        index_for_trainset = []
        for j in range(args.mfold):
            if j != i:
                index_for_trainset.append(mfold_indexs[j])
        index_for_trainset = np.concatenate(index_for_trainset)

        if training_strategy:
            # Update 2021/8/6: New train and test index
            # Swap index_for_trainset and index_for_testset, add fixed index to new train index list
            index_for_trainset_ = index_for_testset
            index_for_testset = index_for_trainset
            index_for_trainset = np.concatenate((index_global, index_for_trainset_))

        mol_features_test = mol_features_all[index_for_testset] if mol_features_all is not None else None
        ground_truths_test = ground_truths_all[index_for_testset]
        SMILES_test = SMILES_all[index_for_testset]
        IDs_test = IDs_all[index_for_testset] if IDs_all is not None else None
        sample_weights_test = sample_weights_all[index_for_testset] if sample_weights_all is not None else None
        fold_args.testset = {'features': mol_features_test,
                             'labels': ground_truths_test,
                             'SMILESs': SMILES_test,
                             'IDs': IDs_test,
                             'sample_weights': sample_weights_test}

        mol_features_train = mol_features_all[index_for_trainset] if mol_features_all is not None else None
        ground_truths_train = ground_truths_all[index_for_trainset]
        SMILES_train = SMILES_all[index_for_trainset]
        IDs_train = IDs_all[index_for_trainset] if IDs_all is not None else None
        sample_weights_train = sample_weights_all[index_for_trainset] if sample_weights_all is not None else None
        fold_args.trainset = {'features': mol_features_train,
                              'labels': ground_truths_train,
                              'SMILESs': SMILES_train,
                              'IDs': IDs_train,
                              'sample_weights': sample_weights_train}

        if fold_args.prefix is None:
            fold_args.prefix = 'Fold %d-%d, model_%s' % (i, args.mfold, args.model_ver)
        else:
            fold_args.prefix = 'Fold %d-%d, ' % (i, args.mfold) + fold_args.prefix
        trainlog = train(fold_args)
        gpickle.dump(trainlog, os.path.join(args.save_root_folder, 'trainlog_mfold_%d-%d.gpkl' % (i, args.mfold)))

    # --- multi-fold variance visualization ---#
    trainlog_files = [x.name for x in os.scandir(args.save_root_folder) if
                      x.is_file() and x.name.startswith('trainlog')]
    if len(trainlog_files) == args.mfold:
        mfold_train_done = True
    else:
        mfold_train_done = False
    if mfold_train_done:
        trainlogs, best_metrics, metric_title = plot_m_variance(args.save_root_folder, prefix='trainlog_mfold')
        print('%s: ave = %0.4f, min = %0.4f, max = %0.4f, std = %0.4f' % (
        metric_title, np.mean(best_metrics), np.min(best_metrics), np.max(best_metrics), np.std(best_metrics)))


# --- grid search ---#
def grid_search_train(args):
    import grid_search, optuna
    from functools import partial
    time_stamp = get_time_stamp()
    warnings.filterwarnings("ignore",
                            message="Choices for a categorical distribution should be a tuple of None, bool, int, float and str for persistent storage")
    if args.mfold is not None:
        raise ValueError('-mfold and -grid_search should not be enabled simultaneously')

    args.save_model_per_epoch_num = args.max_epoch_num + 1
    if args.save_root_folder is None:
        args.save_root_folder = os.path.join(os.getcwd(),
                                             'train_results/grid_search_%s@%s' % (args.model_ver, get_time_stamp()))
    if not os.path.exists(args.save_root_folder):
        os.mkdir(args.save_root_folder)
    if args.rdb is None:
        args.rdb = os.path.join(args.save_root_folder, 'grid_search.db')
    trial_objective_func = getattr(grid_search, 'objective_%s' % args.model_ver)
    trial_objective_func = partial(trial_objective_func, args=args)
    study = optuna.create_study(direction='maximize', storage='sqlite:///' + args.rdb, load_if_exists=True,
                                study_name='grid_search')
    print('Start grid searching for hyperparameter tuning ...')
    time0 = time.time()
    study.optimize(trial_objective_func, n_trials=args.grid_search)
    time1 = time.time()
    print('Number of finished trials = %d, time cost = %0.2fhours' % (len(study.trials), (time1 - time0) / 3600))
    print('Best trial got:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
    with open(os.path.join(args.save_root_folder, 'best_trial_%s@%s.json' % (args.model_ver, time_stamp)),
              encoding='utf8', mode='wt') as f:
        json.dump(trial.params, f, ensure_ascii=False, indent=2)
    gpickle.dump(study, os.path.join(args.save_root_folder, 'study_%s@%s.gpkl' % (args.model_ver, time_stamp)))


# --- bundle training ---#
def multi_run_train(args):
    mrun_datasets = None
    if args.srcfolder is not None:
        if args.trainset is not None or args.testset is not None:
            raise ValueError('MRUN: srcfolder and (trainset, testset) should not be specified simultaneously')
        mrun_datasets = []
        if os.path.exists(args.srcfolder):
            existing_mrun_folders = [x.name for x in os.scandir(args.srcfolder) if
                                     x.is_dir() and x.name.startswith('[MRUN')]
            # ---- exception handling ----#
            existing_mrun_indexs = set()
            for item in existing_mrun_folders:
                endidx = item.find(']')
                idx = int(item[5:endidx].split(',')[0].split('-')[0])
                existing_mrun_indexs.add(idx)
            if len(existing_mrun_folders) != args.mrun:
                raise ValueError('`mrun` = %d whereas %d runs located under srcfolder = %s' % (
                args.mrun, len(existing_mrun_folders), args.srcfolder))
            for i in range(args.mrun):
                if i not in existing_mrun_indexs:
                    raise ValueError('%d-th run folder not found under srcfolder = %s' % (i, args.srcfolder))
            # ---- retrieve trainset & testset files ----#
            for mrun_folder in existing_mrun_folders:
                trainset_file, testset_file = _retrieve_dataset_from_save_folder(
                    os.path.join(args.srcfolder, mrun_folder))
                mrun_datasets.append([trainset_file, testset_file])
            print('Total %d mrun datasests located under %s' % (len(mrun_datasets), args.srcfolder))
        else:
            raise ValueError('MRUN: srcfolder not exist: %s' % args.srcfolder)

    if args.save_root_folder is None:
        if args.prefix is not None:
            args.save_root_folder = os.path.join(os.getcwd(), 'train_results/[%s]_%d_run_%s_model_%s@%s' % (
            args.prefix, args.mrun, args.task, args.model_ver, get_time_stamp()))
        else:
            args.save_root_folder = os.path.join(os.getcwd(), 'train_results/%d_run_%s_model_%s@%s' % (
            args.mrun, args.task, args.model_ver, get_time_stamp()))
    if not os.path.exists(args.save_root_folder):
        os.makedirs(args.save_root_folder)

    for i in range(args.mrun):
        onerun_args = copy.deepcopy(args)
        if onerun_args.prefix is None:
            onerun_args.prefix = 'MRUN %d-%d, model_%s' % (i, args.mrun, args.model_ver)
        else:
            onerun_args.prefix = 'MRUN %d-%d, ' % (i, args.mrun) + onerun_args.prefix
        if mrun_datasets is not None:
            onerun_args.trainset, onerun_args.testset = mrun_datasets[i]
            onerun_args.srcfolder = None

        # ---- scan for existing run folders ----#
        existing_mrun_folders = [x.name for x in os.scandir(args.save_root_folder) if
                                 x.is_dir() and x.name.startswith('[MRUN')]
        existing_mfold_indexs = set()
        for item in existing_mrun_folders:
            endidx = item.find(']')
            idx = int(item[5:endidx].split(',')[0].split('-')[0])
            existing_mfold_indexs.add(idx)
        if i in existing_mfold_indexs:
            continue

        trainlog = train(onerun_args)
        gpickle.dump(trainlog, os.path.join(args.save_root_folder, 'trainlog_mrun_%d-%d.gpkl' % (i, args.mrun)))
    trainlog_files = [x.name for x in os.scandir(args.save_root_folder) if
                      x.is_file() and x.name.startswith('trainlog_mrun_')]
    if len(trainlog_files) == args.mrun:
        mrun_train_done = True
    else:
        mrun_train_done = False
    if mrun_train_done:
        trainlogs, best_metrics, metric_title = plot_m_variance(args.save_root_folder, prefix='trainlog_mrun')
        print('%s: ave = %0.4f, min = %0.4f, max = %0.4f, std = %0.4f' % (
        metric_title, np.mean(best_metrics), np.min(best_metrics), np.max(best_metrics), np.std(best_metrics)))


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-device', default='-1', type=str,
                           help='device, -1=CPU, >=0=GPU, use "," as seperator for multiple devices')
    argparser.add_argument('-model_file', default=None, type=str)
    argparser.add_argument('-model_ver', default='4v4', type=str)
    argparser.add_argument('-class_num', default=2, type=int, help='for regression task, set class_num to 1')
    argparser.add_argument('-pred_threshold', default=0.5, type=float,  help='set prediction threshold for NEGETIVE class, the positive class is automaticly set: (1- pred_threshold )')
    argparser.add_argument('-optimizer', default='adam', type=str, help='sgd|adam|adadelta')
    argparser.add_argument('-trainset', type=str, help='path of trainset .gpkl file')
    argparser.add_argument('-num_local_sample', default=48, type=int, help="Number of local samples in model_file.")  # Update: 2021/8/10, by qi.liu
    argparser.add_argument('-training_strategy', default=None, type=str, help="Default None. ")  # Update: 2021/8/11 by qi.liu
    argparser.add_argument('-testset', default=None, type=str,
                           help='path of testset .gpkl file or partition ratio value in range (0, 1.0)')
    argparser.add_argument('-batch_size', default=1, type=int)
    argparser.add_argument('-batch_size_min', default=1, type=int)
    argparser.add_argument('-batch_size_test', default=1, type=int)
    argparser.add_argument('-ER_start_test', default=50.0, type=float, help='for classification task')
    argparser.add_argument('-RMSE_start_test', default=4.0, type=float, help='for regression task')
    argparser.add_argument('-max_epoch_num', default=500, type=int)
    argparser.add_argument('-save_model_per_epoch_num', default=100000.0, type=float)
    argparser.add_argument('-test_model_per_epoch_num', default=1.0, type=float)
    argparser.add_argument('-save_root_folder', default=None, type=str)
    argparser.add_argument('-save_folder', default=None, type=str)
    argparser.add_argument('-train_loader_worker', default=1, type=int)
    argparser.add_argument('-test_loader_worker', default=1, type=int)
    argparser.add_argument('-use_multiprocessing', default='true', type=str)
    argparser.add_argument('-balance_method', default=0, type=int, help='refer to README for details')
    argparser.add_argument('-weighting_method', default=0, type=int, help='refer to README for details')
    argparser.add_argument('-prefix', default=None, type=str,
                           help='set this as stamp for differentiating multiple runs')
    argparser.add_argument('-config', default=None, type=str, help='param configuration set')
    argparser.add_argument('-select_by_aupr', default='true', type=str,
                           help='select model with AuPR metric, for classification task')
    argparser.add_argument('-select_by_corr', default='true', type=str,
                           help='select model with correlation metric, for regression task')
    argparser.add_argument('-silent', action='store_true', help='disable screen output')
    argparser.add_argument('-verbose', default=0, type=int, help='verbose level for screen output')
    argparser.add_argument('-alpha', default=(0.01, 0.02, 0.05, 0.1), nargs='+', type=float,
                           help='use blank for multiple input seperator')
    argparser.add_argument('-alpha_topk', default=(100, 200, 500, 1000), nargs='+', type=float,
                           help='use blank for multiple input seperator')
    argparser.add_argument('-score_levels', default=(0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5), nargs='+', type=float,
                           help='use blank for multiple input seperator')
    # argparser.add_argument('-mfold', default=None, type=str, help='multi-fold cross validation, an integer or path of `mfold_indexs.gpkl` file')
    argparser.add_argument('-mfold', default="5", type=str, help='multi-fold cross validation, an integer or path of `mfold_indexs.gpkl` file')
    argparser.add_argument('-mrun', default=1, type=int,
                           help='multiple random runs, integer, will be disabled if multi-fold validation or grid search is enabled')
    argparser.add_argument('-early_stop', default=50, type=int,
                           help='stop training after n consecutive tests without better metric got, 0 = disabled')
    argparser.add_argument('-grid_search', default=0, type=int,
                           help='maximum trial number for hyperparameter search, 0 = disabled')
    argparser.add_argument('-group_by_scaffold', action='store_true',
                           help='set this flag to turn on `split by scaffold` for trainset')
    argparser.add_argument('-srcfolder', default=None, type=str, help='source folder for re-run')

    # --- for experiments ---#
    # argparser.add_argument('-embedding_file', default='embedding_from_GE_model_0.gpkl', type=str, help='path of pretrained embedding weight')
    argparser.add_argument('-embedding_file', default=None, type=str, help='path of pretrained embedding weight')
    argparser.add_argument('-train_partial', default=1.0, type=float,
                           help='percent of trainset would be actually used for training')
    argparser.add_argument('-pretrained', default=0, type=int, help='freeze pretrained weights for given epochs')
    argparser.add_argument('-load_weights_strict', default='true', type=str,
                           help='whether load model weights by strictly checking')
    argparser.add_argument('-rdb', default=None, type=str, help='RDB address for parallel grid search')
    argparser.add_argument('-reset_classifier', default='false', type=str, help='for feature generalization experiment')
    argparser.add_argument('-lr', default=None, type=float, help='change default optimizer learning rate')
    argparser.add_argument('-upb', default=1, type=int, help='update model per batch num')
    argparser.add_argument('-accum_loss_num',default=256, type=int, help='Use accumulated loss, gradients backwards after reaching the set accumalated loss number')
    args = argparser.parse_args()

    args.use_multiprocessing = args.use_multiprocessing.lower() in {'true', 'yes'}
    args.select_by_aupr = args.select_by_aupr.lower() in {'true', 'yes'}
    args.select_by_corr = args.select_by_corr.lower() in {'true', 'yes'}
    args.load_weights_strict = args.load_weights_strict.lower() in {'true', 'yes'}
    args.reset_classifier = args.reset_classifier.lower() in {'true', 'yes'}
    args.task = 'regression' if args.class_num == 1 else 'classification'
    if args.grid_search > 0 and args.mfold is not None:
        raise ValueError('-mfold and -grid_search should not be enabled simultaneously')
    if args.srcfolder is not None and (args.trainset is not None or args.testset is not None):
        raise ValueError('srcfolder and (trainset, testset) should not be specified simultaneously')

    # --- unexposed args ---#
    args.anneal_dropouts = [(0.0, 0), (0.0, 0.5)]
    # args.weight_decay = 0.000001
    # args.lr = 1e-2
    # args.upb = 4
    # args.csr = True

    # --- plain training ---#
    if args.mfold is None and args.grid_search == 0:
        if args.mrun <= 1:
            trainlog = train(args)
        # ---- mrun ----#
        else:
            multi_run_train(args)

    # --- multi-fold cross validation ---#
    elif int(args.mfold) >= 1:
        multi_fold_train(args)
    
    elif int(args.mfold) == 0:
        trainlog = train(args)

    # --- grid search ---#
    elif args.grid_search > 0:
        grid_search_train(args)

    print('All done~')
