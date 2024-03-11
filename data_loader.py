# coding:utf-8
"""
Data loader for different experiments
Created  :   7, 12, 2019
Revised  :   7, 12, 2019
Author   :  David Leon (dawei.leng@ghddi.org) Pascal Guo (jinjiang.guo@ghddi.org) 
All rights reserved
-------------------------------------------------------------------------------
"""
__author__ = 'dawei.leng, jinjiang.guo'

import sys, os, time
# from os.path import dirname, abspath
# d = dirname(dirname(abspath(__file__)))
# sys.path.insert(0, d)

import numpy as np, psutil, multiprocessing, threading
from rdkit import Chem
from ligand_based_VS_data_preprocessing import canonicalize_molecule, canonicalize_smiles, convert_SMILES_to_graph
from featurizer import ConvMol
from functools import partial
from graph_utils import calc_degree_for_center_nodes


def scale_membership_by_degree(membership_v, padded_neighbors):
    """

    :param membership_v: original membership values, (node_num,)
    :param padded_neighbors: as returned by `load_batch_data_2` & `load_batch_data_2_1`, (node_num, max_degree)
    :return: scaled membership value
    """
    # print('membership scaled')
    weight = (padded_neighbors > 0).sum(axis=1)  # (node_num, )
    membership_v = membership_v * weight
    return membership_v.astype(np.float32)

def load_batch_data_4v3_1(features, batch_sample_idx, add_self_connection=False, return_format='neighbor_op',
                          aux_data_list=None):
    """
    Corresponding to model_2, input is atom feature sequence computed via DeepChem package
    :param features:
    :param labels:
    :param batch_sample_idx:
    :param add_self_connection: whether append self-connection in adjacency matrix
    :return:
    """
    batch_size = len(batch_sample_idx)
    edges = []
    start_idxs = [0]
    nodes_all = []
    max_degree = 0
    max_node_num = 0
    neighbor_list_all = []
    node_membership_list_all = []
    membership = []
    for i in range(batch_size):
        nodes, neighbors_list = features[batch_sample_idx[i]]
        nodes_all.append(nodes)
        n = nodes.shape[0]
        node_membership_list_all.append([start_idxs[i] + idx for idx in range(n)])
        membership.extend([i for _ in range(n)])
        start_idxs.append(n + start_idxs[i])
        edge_list = []
        for j in range(n):
            for k in neighbors_list[j]:
                edge_list.append([j + start_idxs[i], k + start_idxs[i]])
            if add_self_connection:
                if [j + start_idxs[i], j + start_idxs[i]] not in edge_list:
                    edge_list.append([j + start_idxs[i], j + start_idxs[i]])
        edges.append(edge_list)
        neighbor_list = [[] for _ in range(n)]
        for edge in edge_list:
            neighbor_list[edge[0] - start_idxs[i]].append(edge[1])
        neighbor_list_all.extend(neighbor_list)
        neighbors_num = [len(neighbor_list[i]) for i in range(n)]
        max_degree = max(max_degree, max(neighbors_num))
        max_node_num = max(max_node_num, n)

    X = np.concatenate(nodes_all, axis=0).astype(np.float32)
    batch_data = [X]

    if return_format == 'neighbor_op':

        for neighbors in neighbor_list_all:
            neighbors.extend([-1] * (max_degree - len(neighbors)))
        for node_membership_list in node_membership_list_all:
            node_membership_list.extend([-1] * (max_node_num - len(node_membership_list)))

        padded_neighbors = np.array(neighbor_list_all, dtype=np.int64)  # (node_num, max_degree_in_batch), int64
        padded_membership = np.array(node_membership_list_all,
                                     dtype=np.int64)  # (batch_size, max_node_num_in_batch), int64
        batch_data.extend([padded_neighbors, padded_membership])

    else:  # return_format = 'scatter'
        edges = np.concatenate(edges, axis=0).transpose().astype(
            np.int64)  # (2, n_pair), each column denotes an edge, from node i to node j, int64
        membership = np.array(membership, dtype=np.int64)  # (node_num,)
        batch_data.extend([edges, membership])

    if aux_data_list is not None:
        for item in aux_data_list:
            batch_data.append(item[batch_sample_idx])

    return batch_data


def load_batch_data_4v3(strings, batch_sample_idx, atom_dict, add_self_connection=False, return_format='neighbor_op',
                        aux_data_list=None):
    """
    Corresponding to model_4v3, input is SMILES strings
    :param strings: SMILES strings
    :param batch_sample_idx:
    :param atom_dict:
    :param add_self_connection: whether append self-connection in adjacency matrix
    :param aux_data_list: list of auxiliary data
    :param return_format: {'neighbor_op' | 'scatter'}, specify return format suiting for neighbor_op or torch-scatter.
    :return: X, padded_neighbors, padded_membership, aux_data_list_batch if return_format = 'neighbor_op'
             X, edges, membership, aux_data_list_batch                   if return_format = 'scatter'
             X: (node_num,), index of each node according to `atom_dict`, int64
             padded_neighbors: (node_num, max_degree), int64, each row represents the 1-hop neighbors for node_i; `-1` are padded
                               to indicate invalid values.
             padded_membership: (batch_num, node_num), int64, each row contains the node indices of the i_th graph, -1` are padded
                               to indicate invalid values.
             edges: (2, edge_num), int64, each column in format of (neighbor_node, center_node)
             membership: (node_num,), int64, representing to which graph the i_th node belongs
    """
    batch_size = len(batch_sample_idx)
    tokenized_sequences = []
    edges = []
    start_idxs = [0]
    max_degree = 0
    max_node_num = 0
    neighbor_list_all = []
    node_membership_list_all = []
    membership = []
    for i in range(batch_size):
        s = strings[batch_sample_idx[i]]
        molecule = Chem.MolFromSmiles(s)
        molecule = canonicalize_molecule(molecule)
        tokenized_seq = []
        for atom in molecule.GetAtoms():
            if atom.GetSymbol() not in atom_dict:
                if '<unk>' in atom_dict:
                    tokenized_seq.append(atom_dict['<unk>'])
                else:
                    tokenized_seq.append(len(atom_dict))  # OOV
            else:
                tokenized_seq.append(atom_dict[atom.GetSymbol()])
        tokenized_sequences.extend(tokenized_seq)
        n = len(tokenized_seq)
        node_membership_list_all.append([start_idxs[i] + idx for idx in range(n)])
        membership.extend([i for _ in range(n)])
        start_idxs.append(n + start_idxs[i])
        edge_list = [(b.GetBeginAtomIdx() + start_idxs[i], b.GetEndAtomIdx() + start_idxs[i]) for b in
                     molecule.GetBonds()]
        edge_list_reverse = [(j, i) for (i, j) in edge_list]
        edges.append(edge_list)
        edges.append(edge_list_reverse)  # add symmetric edge
        if add_self_connection:
            edges.append([(j + start_idxs[i], j + start_idxs[i]) for j in range(n)])
        neighbor_list = [[] for _ in range(n)]
        for edge in edge_list + edge_list_reverse:
            neighbor_list[edge[0] - start_idxs[i]].append(edge[1])
        neighbor_list_all.extend(neighbor_list)
        neighbors_num = [len(neighbor_list[i]) for i in range(n)]
        max_degree = max(max_degree, max(neighbors_num))
        max_node_num = max(max_node_num, n)

    X = np.array(tokenized_sequences, dtype=np.int64)  # (node_num,), index of each node, int64
    batch_data = [X]

    if return_format == 'neighbor_op':
        for neighbors in neighbor_list_all:
            neighbors.extend([-1] * (max_degree - len(neighbors)))
        for node_membership_list in node_membership_list_all:
            node_membership_list.extend([-1] * (max_node_num - len(node_membership_list)))

        padded_neighbors = np.array(neighbor_list_all, dtype=np.int64)  # (node_num, max_degree_in_batch), int64
        padded_membership = np.array(node_membership_list_all,
                                     dtype=np.int64)  # (batch_size, max_node_num_in_batch), int64
        batch_data.extend([padded_neighbors, padded_membership])
    else:  # return_format = 'scatter'
        edges = np.concatenate(edges, axis=0).transpose().astype(
            np.int64)  # (2, n_pair), each column denotes an edge, from node i to node j, int64
        membership = np.array(membership, dtype=np.int64)  # (node_num,)
        batch_data.extend([edges, membership])

    if aux_data_list is not None:
        for item in aux_data_list:
            batch_data.append(item[batch_sample_idx])

    return batch_data


def convert_to_degree_wise_format(X, edges, membership):
    """
    Convert loaded data to degree-wise format.
    This function is computation intensive.
    :param X, edges, membership as returned by load_batch_data_4v3/4v4
    :return X, edges, membership, degree_slices as required by model_4v4
    """
    time0 = time.time()
    node_degrees = calc_degree_for_center_nodes(edges.T).astype(np.int64)
    time1 = time.time()
    node_num = len(node_degrees)
    assert node_num == X.shape[0]
    idx_sorted = np.argsort(node_degrees)
    node_degrees = node_degrees[idx_sorted]  # (node_num,)
    X = X[idx_sorted]  # (node_num,)
    membership = membership[idx_sorted]  # (node_num,)
    time2 = time.time()
    idx_map = dict()
    for idx_new, idx_old in enumerate(idx_sorted):
        idx_map[idx_old] = idx_new
    idx_map = [idx_map[key] for key in range(node_num)]
    idx_map = np.array(idx_map).astype(np.int64)
    edges = idx_map[edges]
    time3 = time.time()
    # todo: codes between time3 and time4 need accelerated
    min_degree, max_degree = node_degrees.min(), node_degrees.max()
    degree_slices = np.zeros((max_degree + 1, 2), dtype=np.int64)
    start_idx = 0
    edges_degree_slices = []
    edge_num = edges.shape[1]
    node_idx_range = np.arange(node_num)
    for degree in range(min_degree, max_degree + 1):
        mask = node_degrees == degree
        node_num_degree = mask.sum()
        edge_num_degree = node_num_degree * max(degree, 1)

        degree_slices[degree, :] = [start_idx, start_idx + edge_num_degree]
        if degree > 0:
            start_idx += edge_num_degree
            node_idxs_degree = node_idx_range[mask]
            for idx in node_idxs_degree:
                mask_idx = edges[1, :] == idx
                if any(mask_idx):
                    edges_degree_slices.append(edges[:, mask_idx])
    time4 = time.time()
    edges = np.concatenate(edges_degree_slices, axis=1)
    time5 = time.time()
    # print('node degree calc time =', time1-time0)
    # print('sort time =', time2-time1)
    # print('edge rename time =', time3-time2)
    # print('degree_slice time =', time4-time3)
    # print('edge sort time =', time5-time4)
    assert edges.shape[1] == edge_num
    return X, edges, membership, degree_slices


def load_batch_data_4v4(samples, batch_sample_idx, atom_dict=None, add_self_connection=False, degree_wise=False,
                        aux_data_list=None, csr_format=False):
    """
    batch data loader for model_4v4
    :param samples: list of SMILES strings or returns from `convert_SMILES_to_graph()` or mixture of them
    :param batch_sample_idx:
    :param atom_dict: required when samples contain SMILES string
    :param add_self_connection: whether append self-connection in adjacency matrix
    :param degree_wise: whether in degree-wise format
    :param aux_data_list: list of auxiliary data
    :return: X, edges, membership, degree_slices (optional), *aux_data (optional)
             X: (node_num,), index of each node according to `atom_dict`, int64
             edges: (2, edge_num), int64, each column in format of (neighbor_node, center_node)
             membership: (node_num,), int64, representing to which graph the i_th node belongs
             degree_slices: (max_degree_in_batch+1, 2), each row in format of (start_idx, end_idx), in which '*_idx' corresponds
                            to edges indices; i.e., each row is the span of edges whose center node is of the same degree,
                            returned only when `degree_wise` = True
             *aux_data: list of auxiliary data organized in one batch
    """
    batch_size = len(batch_sample_idx)
    tokenized_sequences = []
    edges = []
    start_idxs = [0]
    membership = []
    for i in range(batch_size):
        s = samples[batch_sample_idx[i]]
        if isinstance(s, str):
            xs, es = convert_SMILES_to_graph(SMILES=s, atom_dict=atom_dict)
        else:
            xs, es = s
        tokenized_sequences.extend(xs)
        n = len(xs)
        membership.extend([i for _ in range(n)])
        start_idxs.append(n + start_idxs[i])
        edges.extend([(node_n + start_idxs[i], node_c + start_idxs[i]) for (node_n, node_c) in es])
        if add_self_connection:
            edges.extend([(j + start_idxs[i], j + start_idxs[i]) for j in range(n)])

    X = np.array(tokenized_sequences, dtype=np.int64)  # (node_num,), index of each node, int64
    edges = np.array(edges).transpose().astype(
        np.int64)  # (2, n_pair), each column denotes an edge, from node i to node j, int64
    membership = np.array(membership, dtype=np.int64)  # (node_num,)                        # (node_num,)

    batch_data = [X, edges, membership]

    if csr_format:
        idxs = np.argsort(edges[1, :])
        edges_sorted = edges[:, idxs]
        index_csr = edges_sorted[1, :]
        idxpt = [0]
        idxv = 0
        L = index_csr.shape[0]
        for i in range(L):
            if index_csr[i] == idxv:
                continue
            else:
                idxv0 = idxv
                for k in range(idxv0, index_csr[i]):
                    idxpt.append(i)
                    idxv += 1
        idxpt.append(L)
        idxpt = np.array(idxpt, dtype=np.int64)
        batch_data.append(idxpt)
        batch_data[1] = edges_sorted

    if degree_wise:
        X, edges, membership, degree_slices = convert_to_degree_wise_format(X, edges, membership)
        batch_data.append(degree_slices)
    if aux_data_list is not None:
        for item in aux_data_list:
            batch_data.append(item[batch_sample_idx])

    return batch_data

def load_batch_data_4v4_multifeats(samples, batch_sample_idx, atom_dict=None, add_self_connection=False, degree_wise=False,
                        aux_data_list=None, csr_format=False):
    """
    batch data loader for model_4v4
    :param samples: list of SMILES strings or returns from `convert_SMILES_to_graph()` or mixture of them
    :param batch_sample_idx:
    :param atom_dict: required when samples contain SMILES string
    :param add_self_connection: whether append self-connection in adjacency matrix
    :param degree_wise: whether in degree-wise format
    :param aux_data_list: list of auxiliary data
    :return: X, edges, membership, degree_slices (optional), *aux_data (optional)
             X: (node_num,), index of each node according to `atom_dict`, int64
             edges: (2, edge_num), int64, each column in format of (neighbor_node, center_node)
             membership: (node_num,), int64, representing to which graph the i_th node belongs
             degree_slices: (max_degree_in_batch+1, 2), each row in format of (start_idx, end_idx), in which '*_idx' corresponds
                            to edges indices; i.e., each row is the span of edges whose center node is of the same degree,
                            returned only when `degree_wise` = True
             *aux_data: list of auxiliary data organized in one batch
    """
    batch_size = len(batch_sample_idx)
    tokenized_sequences = []
    edges = []
    start_idxs = [0]
    membership = []
    for i in range(batch_size):
        s = samples[batch_sample_idx[i]]
        if isinstance(s, str):
            xs, es = convert_SMILES_to_graph(SMILES=s, atom_dict=atom_dict)
        else:
            xs, es = s
        tokenized_sequences.extend(xs)
        n = len(xs)
        membership.extend([i for _ in range(n)])
        start_idxs.append(n + start_idxs[i])
        edges.extend([(node_n + start_idxs[i], node_c + start_idxs[i]) for (node_n, node_c) in es])
        if add_self_connection:
            edges.extend([(j + start_idxs[i], j + start_idxs[i]) for j in range(n)])

    X = np.array(tokenized_sequences, dtype=np.int64)  # (node_num,), index of each node, int64
    edges = np.array(edges).transpose().astype(
        np.int64)  # (2, n_pair), each column denotes an edge, from node i to node j, int64
    membership = np.array(membership, dtype=np.int64)  # (node_num,)                        # (node_num,)

    batch_data = [X, edges, membership]

    if csr_format:
        idxs = np.argsort(edges[1, :])
        edges_sorted = edges[:, idxs]
        index_csr = edges_sorted[1, :]
        idxpt = [0]
        idxv = 0
        L = index_csr.shape[0]
        for i in range(L):
            if index_csr[i] == idxv:
                continue
            else:
                idxv0 = idxv
                for k in range(idxv0, index_csr[i]):
                    idxpt.append(i)
                    idxv += 1
        idxpt.append(L)
        idxpt = np.array(idxpt, dtype=np.int64)
        batch_data.append(idxpt)
        batch_data[1] = edges_sorted
    if degree_wise:
        X, edges, membership, degree_slices = convert_to_degree_wise_format(X, edges, membership)
        batch_data.append(degree_slices)
    if aux_data_list is not None:
        for item in aux_data_list:
            batch_data.append(item[batch_sample_idx])
    return batch_data

def load_batch_data_4v4_1(features, batch_sample_idx, add_self_connection=False, degree_wise=False, aux_data_list=None):
    """
    Corresponding to model_4v4, input is atom feature sequence computed via DeepChem package
    :param features:
    :param labels:
    :param batch_sample_idx:
    :param degree_wise: whether in degree-wise format
    :param add_self_connection: whether append self-connection in adjacency matrix
    :return:
    """
    batch_size = len(batch_sample_idx)

    edges = []
    start_idxs = [0]
    nodes_all = []
    membership = []
    for i in range(batch_size):
        nodes, neighbors_list = features[batch_sample_idx[i]]
        nodes_all.append(nodes)
        n = nodes.shape[0]
        membership.extend([i for _ in range(n)])
        start_idxs.append(n + start_idxs[i])
        edge_list = []
        for j in range(n):
            for k in neighbors_list[j]:
                edge_list.append([j + start_idxs[i], k + start_idxs[i]])
            if add_self_connection:
                if [j + start_idxs[i], j + start_idxs[i]] not in edge_list:
                    edge_list.append([j + start_idxs[i], j + start_idxs[i]])
        edges.append(edge_list)

    X = np.concatenate(nodes_all, axis=0).astype(np.float32)
    edges = np.concatenate(edges, axis=0).transpose().astype(
        np.int64)  # (2, n_pair), each column denotes an edge, from node i to node j, int64
    membership = np.array(membership, dtype=np.int64)  # (node_num,)

    batch_data = [X, edges, membership]
    if degree_wise:
        X, edges, membership, degree_slices = convert_to_degree_wise_format(X, edges, membership)
        batch_data.append(degree_slices)
    if aux_data_list is not None:
        for item in aux_data_list:
            batch_data.append(item[batch_sample_idx])
    return batch_data


load_batch_data_4v5 = load_batch_data_4v4
load_batch_data_4v5_1 = load_batch_data_4v4_1
load_batch_data_4v6 = load_batch_data_4v4
load_batch_data_4v6_1 = load_batch_data_4v4_1
load_batch_data_4v7 = load_batch_data_4v4
load_batch_data_4v7_1 = load_batch_data_4v4_1
load_batch_data_5 = load_batch_data_4v4
load_batch_data_5_1 = load_batch_data_4v4_1


def feed_sample_batch(batch_data_loader,
                      features,
                      aux_data_list=None,
                      data_queue=None,
                      max_epoch_num=0,
                      batch_size=64, batch_size_min=None,
                      shuffle=False,
                      use_multiprocessing=False,
                      epoch_start_event=None,
                      epoch_done_event=None
                      ):
    me_process = psutil.Process(os.getpid())
    sample_num = features.shape[0]
    if batch_size_min is None:
        batch_size_min = batch_size
    batch_size_min = min([batch_size_min, batch_size])
    done, epoch = False, 0
    while not done:
        if use_multiprocessing:
            if me_process.parent() is None:  # parent process is dead
                raise RuntimeError('Parent process is dead, exiting')
        if epoch_start_event is not None:
            epoch_start_event.wait()
            epoch_start_event.clear()
        if epoch_done_event is not None:
            epoch_done_event.clear()
        if shuffle:
            index = np.random.choice(range(sample_num), size=sample_num, replace=False)
        else:
            index = np.arange(sample_num)
        index = list(index)
        end_idx = 0
        while end_idx < sample_num:
            current_batch_size = np.random.randint(batch_size_min, batch_size + 1)
            start_idx = end_idx
            end_idx = min(start_idx + current_batch_size, sample_num)
            batch_sample_idx = index[start_idx:end_idx]
            batch_data = batch_data_loader(features, batch_sample_idx=batch_sample_idx, aux_data_list=aux_data_list)
            data_queue.put(batch_data)

        if epoch_done_event is not None:
            # time.sleep(3.0)   # most possible jitter time for cross process communication (mp.queue)
            epoch_done_event.set()
        epoch += 1
        if max_epoch_num > 0:
            if epoch >= max_epoch_num:
                done = True


def sync_manager(workers_epoch_start_events, workers_epoch_done_events):
    while 1:
        for event in workers_epoch_done_events:
            event.wait()
            event.clear()
        # data_queue.put(None)   # tell the queue consumer that the epoch is done
        for event in workers_epoch_start_events:
            event.set()


class Data_Loader_Manager(object):
    def __init__(self,
                 batch_data_loader,
                 data_queue,
                 data,
                 shuffle=False,
                 batch_size=128,
                 batch_size_min=None,
                 worker_num=1,
                 use_multiprocessing=False,
                 auto_rewind=0,
                 name=None,
                 ):
        """
        :param auto_rewind: int, 0 means no auto-rewinding, >0 means auto-rewinding for at most n epochs
        """
        super().__init__()
        self.batch_data_loader = batch_data_loader
        self.data_queue = data_queue
        self.data = data
        self.shuffle = shuffle
        self.batch_size_max = batch_size
        self.batch_size_min = self.batch_size_max if batch_size_min is None else batch_size_min
        self.worker_num = worker_num
        self.use_multiprocessing = use_multiprocessing
        self.auto_rewind = auto_rewind
        self.name = name
        self.workers = []
        self.workers_epoch_start_events = []
        self.workers_epoch_done_events = []

        self.batch_size_min = min(self.batch_size_max, self.batch_size_min)

        if use_multiprocessing:
            Worker = multiprocessing.Process
            Event = multiprocessing.Event
        else:
            Worker = threading.Thread
            Event = threading.Event
        sample_num = data[0].shape[0]
        X, *aux_data_list = data  # X.shape = (n_sample, 2), np-array of tuples (nodes, neighbor_list), Y = ground_truth, shape=(n_sample,) / None

        startidx, endidx, idxstep = 0, 0, sample_num // worker_num

        for i in range(worker_num):
            worker_epoch_start_event = Event()
            worker_epoch_done_event = Event()
            worker_epoch_start_event.set()
            worker_epoch_done_event.clear()
            self.workers_epoch_start_events.append(worker_epoch_start_event)
            self.workers_epoch_done_events.append(worker_epoch_done_event)

            startidx = i * idxstep
            if i == worker_num - 1:
                endidx = sample_num
            else:
                endidx = startidx + idxstep
            aux_data_list_worker = None
            if aux_data_list is not None or len(aux_data_list) > 0:
                aux_data_list_worker = []
                for item in aux_data_list:
                    if item is not None:
                        aux_data_list_worker.append(item[startidx:endidx])
            data_proc = Worker(target=feed_sample_batch,
                               args=(batch_data_loader,
                                     X[startidx:endidx], aux_data_list_worker, data_queue,
                                     auto_rewind, batch_size, batch_size_min, shuffle,
                                     use_multiprocessing,
                                     worker_epoch_start_event,
                                     worker_epoch_done_event
                                     ),
                               name='%s_thread_%d' % (name, i))
            data_proc.daemon = True
            self.workers.append(data_proc)

        if auto_rewind > 0:
            sync_manager_proc = Worker(target=sync_manager,
                                       args=(self.workers_epoch_start_events,
                                             self.workers_epoch_done_events,
                                             ),
                                       name='%s_sync_manager' % name)
            sync_manager_proc.daemon = True
            self.workers.append(sync_manager_proc)

        for proc in self.workers:
            proc.start()
            print('%s started' % proc.name)

    def rewind(self):
        for event in self.workers_epoch_done_events:
            event.wait()
            event.clear()
        for event in self.workers_epoch_start_events:
            event.set()

    def close(self):
        for proc in self.workers:
            proc.terminate()
