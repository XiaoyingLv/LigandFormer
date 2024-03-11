# coding:utf-8
"""
Virtual screening evaluation / prediction
Created  :   6, 25, 2019
Revised  :   2,  1, 2020
Author   :  David Leon (dawei.leng@ghddi.org)
All rights reserved
-------------------------------------------------------------------------------
"""
__author__ = 'dawei.leng'

import os
from os.path import join as pjoin
import sys
import warnings
import time
import multiprocessing
import queue

import torch
from torch.nn.functional import softmax
from util.convergence_plot import visualize_atom_attention
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from pytorch_ext.util import get_local_ip, get_time_stamp, gpickle

import metrics
import data_loader
import config as experiment_configs
from ligand_based_VS_model import Model_Agent

torch.set_num_threads(2)
try:
    local_ip = get_local_ip()
except:
    local_ip = 'None'
time_stamp = get_time_stamp()


def compute_confusion_matrix(predicted_results, groundtruths, class_num):
    """
    """
    confusion_matrix = np.zeros(shape=(class_num, class_num))
    for groundtruth, predicted_result in zip(groundtruths, predicted_results):
        confusion_matrix[groundtruth, predicted_result] += 1
    confusionmatrix_normlized = np.copy(confusion_matrix)
    for i in range(class_num):
        s = np.sum(confusionmatrix_normlized[i, :])
        if s > 0:
            confusionmatrix_normlized[i, :] /= s
    return confusion_matrix, confusionmatrix_normlized


def confusion_analysis(confusion_matrix):
    err_num_per_class = confusion_matrix.sum(axis=1) - confusion_matrix.diagonal()
    most_erroneous_class_index = np.argsort(err_num_per_class)[::-1]
    most_confused_class_index = np.argsort(confusion_matrix, axis=1)[:, ::-1]
    return most_erroneous_class_index, most_confused_class_index, err_num_per_class


class TestLog(object):
    def __init__(self):
        super().__init__()


def model_evaluate(args):
    # --- public paras ---#
    device = args.device
    model_file = args.model_file
    model_ver = args.model_ver
    class_num = args.class_num
    pred_threshold = args.pred_threshold
    testset = args.testset
    batch_size = args.batchsize
    save_root_folder = args.save_root_folder
    use_multiprocessing = args.use_multiprocessing
    result_prefix = args.result_prefix
    config_set = args.config
    viz_att = args.viz_attention
    alpha = (0.01, 0.02, 0.05, 0.1)
    test_loader_worker = 1  # fixed to 1
    testlog = TestLog()

    # --- (1) setup device ---#
    if device < 0:
        print('Using CPU for evaluation')
        device = torch.device('cpu')
    else:
        print('Using CUDA%d for evaluation' % device)
        device = torch.device('cuda:%d' % device)

    # --- (2) prepare model ---#
    if config_set is None:
        config_set = 'model_%s_config' % model_ver
    config = getattr(experiment_configs, config_set)()
    task = 'regression' if class_num == 1 else 'classification'
    model_agent = Model_Agent(device=device, model_ver=args.model_ver, output_dim=args.class_num, task=task,
                              config=config, model_file=model_file, atom_dict_file=args.atom_dict_file,
                              load_weights_strict=args.load_weights_strict, load_all_state_dicts=True, viz_att=viz_att)
    if model_agent.model_file_md5 is not None:
        print('model weights loaded')
    else:
        raise ValueError('model file is not specified')

    print('local ip = %s, model ver = %s, time_stamp = %s' % (local_ip, model_ver, time_stamp))

    # --- (3) setup data ---#
    testset_data = gpickle.load(testset)
    if isinstance(testset_data, dict):
        mol_features_test = testset_data['features']
        ground_truths_test = testset_data['labels']
    else:
        mol_features_test, ground_truths_test = testset_data[0], testset_data[1]
    if len(testset_data) > 2:
        if isinstance(testset_data, dict):
            SMILES_test = testset_data['SMILESs']
            IDs_test = testset_data['IDs']
        else:
            SMILES_test, IDs_test, *aux_data_list = testset_data[2:]
    else:
        SMILES_test, IDs_test = None, None
    if SMILES_test is None:
        SMILES_test = mol_features_test
    test_sample_num = mol_features_test.shape[0]
    if IDs_test is None:
        IDs_test = list(range(test_sample_num))  # TODO: query the original ID 2021/08/31
    if save_root_folder is None:
        save_root_folder = os.path.join(os.getcwd(), 'evaluation')
    if not os.path.exists(save_root_folder):
        os.makedirs(save_root_folder)
    save_folder = os.path.join(save_root_folder, os.path.splitext(os.path.basename(model_file))[0])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    testlog.save_folder = save_folder
    queue_size = 3  # Set queue_size
    if use_multiprocessing:
        data_container = multiprocessing.Manager().Queue
    else:
        data_container = queue.Queue
    test_data_queue = data_container(queue_size * test_loader_worker)  # TODO
    test_DLM = data_loader.Data_Loader_Manager(batch_data_loader=model_agent.batch_data_loader,
                                               data_queue=test_data_queue,  # TODO:
                                               data=(mol_features_test, ground_truths_test),
                                               shuffle=False,  # TODO:  Decoupling (eg. if shuffle is True, the report will have mistakes)
                                               batch_size=batch_size,
                                               worker_num=test_loader_worker,
                                               use_multiprocessing=use_multiprocessing,
                                               auto_rewind=0,
                                               name='test_DLM')

    # --- (4) evaluation ---#
    print('Evaluating model on test set (size = %d)' % test_sample_num)
    model_agent.model.train(False)
    # for module in (model_agent.model.dense0, model_agent.model.bn0, model_agent.model.dense1, model_agent.model.bn1, model_agent.model.dense2):
    #     #     module.reset_parameters()
    predicted_results, groundtruths = [], []
    test_batch_num = (test_sample_num + batch_size - 1) // batch_size
    test_time0 = time.time()
    tested_sample_num, total_wrong, ER_test = 0.0, 0.0, 100.0
    scorematrix_all = []
    with torch.no_grad():
        while tested_sample_num < test_sample_num:
            time0 = time.time()
            # TODO: To figure out
            batch_test_data = test_data_queue.get()
            time1 = time.time()
            X, Y = batch_test_data[0], batch_test_data[-1]  # TODO: X, Y
            scorematrix, *_ = model_agent.forward(batch_test_data)
            # TODO:  score matrix , graphs
            # ---- metric calculation ----#
            if task == 'regression':
                scorematrix_all.append(scorematrix.cpu().detach().numpy())
                RMSE_test = np.sqrt(((scorematrix.squeeze(1).cpu().detach().numpy() - Y) ** 2).mean())
                std_AE_test = np.std(np.abs(scorematrix.squeeze(1).cpu().detach().numpy() - Y))
                prediction = scorematrix.squeeze(1).cpu().detach().numpy()

                batch_sample_num = Y.shape[0]
                tested_sample_num += batch_sample_num
                for i in range(batch_sample_num):
                    predicted_results.append(prediction[i])
                    groundtruths.append(Y[i])
                test_time1 = time.time()
                progress = tested_sample_num / test_sample_num
                print('RMSE_test = %0.4f, std_AE_test = %0.4f, progress = %0.2f%%, time remained = %0.2fmins' % (
                    RMSE_test, std_AE_test, progress * 100, (test_time1 - test_time0) / 60 * (1 - progress) / progress))
            else:
                scorematrix = softmax(scorematrix, dim=1)
                scorematrix_all.append(scorematrix.cpu().detach().numpy())
                # best_ps, best_labels = torch.max(scorematrix, dim=1)
                score_mask_neg = torch.unsqueeze((scorematrix[:, 0] >= pred_threshold), 1)  #set threshold negitve
                score_mask_pos = torch.unsqueeze((scorematrix[:, 1] > (1 - pred_threshold)), 1)  #set threshold postive
                score_mask = torch.cat((score_mask_neg, score_mask_pos), dim =1)
                best_labels = torch.argmax(score_mask.long(), dim=1)
                best_ps = torch.masked_select(scorematrix, score_mask)
                prediction = best_labels.cpu().detach().numpy().astype(np.int32)
                prediction_probability = best_ps.cpu().detach().numpy()

                batch_sample_num = Y.shape[0]
                for i in range(batch_sample_num):
                    predicted_results.append([prediction[i], prediction_probability[i]])  # TODO:
                    groundtruths.append(Y[i])
                wrong_sample_num = np.sum(prediction != Y)
                tested_sample_num += batch_sample_num
                total_wrong += wrong_sample_num
                ER_test = total_wrong / tested_sample_num * 100.0
                time2 = time.time()
                data_process_time = time1 - time0
                test_time = time2 - time1
                progress = tested_sample_num / test_sample_num
                print('ER_test = %0.2f, time = %0.2fs(%0.2f|%0.2f), progress = %0.2f%%, time remained = %0.2fmins' % (
                    ER_test, (time2 - time0), test_time, data_process_time, progress * 100,
                    (time2 - test_time0) / 60 * (1 - progress) / progress))
    test_time2 = time.time()
    test_DLM.close()

    test_time_cost = test_time2 - test_time0
    if task == 'regression':
        predicted_results = np.array(predicted_results[:test_sample_num])
        groundtruths = np.array(groundtruths[:test_sample_num], dtype=np.float32)
        pearson_corr_test, pp_value_test = metrics.pearson_corr(predicted_results, groundtruths)
        spearman_corr_test, sp_value_test = metrics.spearman_corr(predicted_results, groundtruths)
        print('Total sampe number: ', len(groundtruths))
        print(
            'Pearson_test = %0.4f, Spearman_test = %0.4f, test time cost = %0.2fmins, speed = %0.2f samples/s @batchsize = %d' % (
                pearson_corr_test, spearman_corr_test, test_time_cost / 60,
                test_batch_num * batch_size / test_time_cost, batch_size))
        for p in multiprocessing.active_children():
            p.terminate()
        gpickle.dump((groundtruths, predicted_results), os.path.join(save_folder, 'predictions.gpkl'))  # TODO:
    else:
        predicted_results = np.array(predicted_results[:test_sample_num])
        groundtruths = np.array(groundtruths[:test_sample_num], dtype=np.int)
        scorematrix = np.concatenate(scorematrix_all, axis=0)
        acc = np.mean(predicted_results[:, 0].astype(np.int) == groundtruths)
        ER_test = 100 - acc * 100
        print('Total sampe number: ', len(groundtruths))
        print('ER_test = %0.2f, test time cost = %0.2fmins, speed = %0.2f samples/s @batchsize = %d' % (
            ER_test, test_time_cost / 60, test_batch_num * batch_size / test_time_cost, batch_size))
        if class_num == 2:
            alpha_acc_list, relative_enrichment_list = [], []
            for a in alpha:
                alpha_acc, positive_ratio, relative_enrichment = metrics.alpha_enrichment(groundtruths,
                                                                                          scorematrix[:, 1], alpha=a)
                print('Alpha_acc = %0.2f%%@%0.1f%%, relative_enrichment = %0.2f, gt_positive_ratio=%0.2f%%' %
                      (alpha_acc * 100, a * 100, relative_enrichment, positive_ratio * 100))
                alpha_acc_list.append(alpha_acc)
                relative_enrichment_list.append(relative_enrichment)
        for p in multiprocessing.active_children():
            p.terminate()
        confusion_matrix, confusionmatrix_normlized = compute_confusion_matrix(predicted_results[:, 0].astype(np.int),
                                                                               groundtruths, class_num)
        most_erroneous_class_index, most_confused_class_index, err_num_per_class = confusion_analysis(confusion_matrix)
        gpickle.dump((predicted_results, groundtruths, confusion_matrix, confusionmatrix_normlized),
                     os.path.join(save_folder, 'confusion_info.gpkl'))
        gpickle.dump((groundtruths, scorematrix), os.path.join(save_folder, 'predictions.gpkl'))
        # --- (4.1) additional metrics ---#
        roc_curves = metrics.roc_curve(groundtruths, scorematrix)
        pr_curves = metrics.pr_curve(groundtruths, scorematrix)
        for i, ftt in enumerate(roc_curves):
            fpr, tpr, th, auroc = ftt
            print('auroc[%d]=' % i, auroc)
            plt.plot(fpr, tpr)
        plt.legend(['roc[%d]' % i for i in range(len(roc_curves))])
        plt.grid(True)
        plt.title('ROC curves')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.savefig(os.path.join(save_folder, result_prefix + 'ROC.png'), dpi=300)
        plt.figure()
        for i, prt in enumerate(pr_curves):
            p, r, th, aupr = prt
            print('aupr[%d]=' % i, aupr)
            plt.plot(r, p)
        plt.legend(['pr[%d]' % i for i in range(len(pr_curves))])
        plt.grid(True)
        plt.title('PR curves')
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.savefig(os.path.join(save_folder, result_prefix + 'PR.png'), dpi=300)

    # --- (5) create report file ---#
    if task == 'regression':
        rpt_file = os.path.join(save_folder, result_prefix + 'RMSE=%0.4f@%s.md' % (RMSE_test, time_stamp))
        logfp = open(rpt_file, mode='wt', encoding='utf8')
        logfp.write('### **Parameters**  \n')
        logfp.write('| param                | value  |  \n')
        logfp.write('| -------------------  | -----: |  \n')
        logfp.write('| Time stamp           |  %s    |  \n' % time_stamp)
        logfp.write('| local_ip             |  %s    |  \n' % local_ip)
        logfp.write('| device               |  %d    |  \n' % args.device)
        logfp.write('| model version        |  %s    |  \n' % model_ver)
        logfp.write('| model file           |  %s    |  \n' % model_file)
        logfp.write('| config               |  %s    |  \n' % config_set)
        logfp.write('| test_set             |  %s    |  \n' % testset)
        logfp.write('| test set size        |  %d    |  \n' % test_sample_num)
        logfp.write('| class_num            |  %d    |  \n' % class_num)
        logfp.write('| batch_size           |  %d    |  \n' % batch_size)
        logfp.write('| RMSE                 |  %0.4f |  \n' % RMSE_test)
        logfp.write('| Pearson              |  %0.4f |  \n' % pearson_corr_test)
        logfp.write('| Spearman             |  %0.4f |  \n' % spearman_corr_test)
        logfp.write('| time cost            |  %0.2fmins |  \n' % (test_time_cost / 60))
        logfp.write('| samples/s            |  %0.2f |  \n' % (test_batch_num * batch_size / test_time_cost))

        logfp.close()
        testlog.rpt_file = rpt_file
    else:
        rpt_file = os.path.join(save_folder, result_prefix + 'ER=%0.2f@%s.md' % (ER_test, time_stamp))
        logfp = open(rpt_file, mode='wt', encoding='utf8')
        logfp.write('### **Parameters**  \n')
        logfp.write('| param                | value  |  \n')
        logfp.write('| -------------------  | -----: |  \n')
        logfp.write('| Time stamp           |  %s    |  \n' % time_stamp)
        logfp.write('| local_ip             |  %s    |  \n' % local_ip)
        logfp.write('| device               |  %d    |  \n' % args.device)
        logfp.write('| model version        |  %s    |  \n' % model_ver)
        logfp.write('| model file           |  %s    |  \n' % model_file)
        logfp.write('| config               |  %s    |  \n' % config_set)
        logfp.write('| test_set             |  %s    |  \n' % testset)
        logfp.write('| test set size        |  %d    |  \n' % test_sample_num)
        logfp.write('| class_num            |  %d    |  \n' % class_num)
        logfp.write('| pred_threshold       |  %d    |  \n' % pred_threshold)
        logfp.write('| batch_size           |  %d    |  \n' % batch_size)
        logfp.write('| ER                   |  %0.2f |  \n' % ER_test)
        logfp.write('| AuROC                | [%s]   |  \n' % ', '.join(['%0.2f' % x[3] for x in roc_curves]))
        logfp.write('| AuPRC                | [%s]   |  \n' % ', '.join(['%0.2f' % x[3] for x in pr_curves]))
        if class_num == 2:
            for a, alpha_acc, relative_enrichment in zip(alpha, alpha_acc_list, relative_enrichment_list):
                logfp.write('| Alpha_enrichment     |  %0.2f\|%0.2f@%0.1f |  \n' % (
                alpha_acc * 100, relative_enrichment, a * 100))
        logfp.write('| time cost            |  %0.2fmins |  \n' % (test_time_cost / 60))
        logfp.write('| samples/s            |  %0.2f |  \n' % (test_batch_num * batch_size / test_time_cost))
        logfp.write(
            '| Note                 |  %s    |  \n' % 'For class number > 2, AuROC and AuPRC are calculated by treating multi-class problem as multi-label problem, so these values are only for weak reference.')

        logfp.write('___   \n')
        logfp.write('### **Testset Class Distribution**   \n')
        logfp.write('| class label      | sample number  | proportion | \n')
        logfp.write('| ---------------  | -------------: | ---------: | \n')
        sample_num_per_class = np.zeros(class_num)
        for i in range(class_num):
            sample_num_per_class[i] = (groundtruths == i).sum()
            logfp.write('| %d  | %d  | %.2f %% |   \n' % (
            i, sample_num_per_class[i], 100.0 * sample_num_per_class[i] / groundtruths.size))

        logfp.write('___   \n')
        logfp.write('### **Metrics**   \n')
        logfp.write('| Class         | Precision | Recall  |   F1  |\n')
        logfp.write('| ------------  | --------: | -----:  |  ---: |\n')
        for i in range(class_num):
            precision = confusion_matrix[i, i] / confusion_matrix[:, i].sum() * 100.0
            recall = confusion_matrix[i, i] / confusion_matrix[i, :].sum() * 100.0
            if precision + recall > 0.0:
                F1 = 2 * precision * recall / (precision + recall)
            else:
                F1 = 0.0
            logfp.write('| %d | %0.2f | %0.2f | %0.2f |\n' % (i, precision, recall, F1))
            print('class_%d: P = %0.2f, R = %0.2f, F1 = %0.2f' % (i, precision, recall, F1))

        logfp.write('___   \n')
        logfp.write('### **Confusion Matrix**   \n')
        logfp.write('| gt\\pr |' + ''.join([' %d |' % i for i in range(class_num)]) + '\n')
        logfp.write('| --- ' + ' '.join(['| ---: ' for i in range(class_num)]) + '\n')
        for i in range(class_num):
            logfp.write('| %d |' % i + ''.join([' %d |' % confusion_matrix[i, j] for j in range(class_num)]) + '\n')

        logfp.write('___   \n')
        logfp.write('### **Error Statistics**   \n')
        logfp.write('| Most Erroneous Class       | Err No.  | Confusion Classes  |  \n')
        logfp.write('| -------------------------  | -------: | -----------------  |  \n')
        # width = int(np.log10(err_num_per_class.max())) + 2
        for i in range(class_num):
            idx = most_erroneous_class_index[i]
            label_final = idx
            if err_num_per_class[idx] > 0:
                # logfp.write('{label} {num:{width}}'.format(label=label_final, num=int(err_num_per_class[idx]), width=width))
                logfp.write('| %s | %d |' % (label_final, err_num_per_class[idx]))
                for j in range(class_num):
                    idx2 = most_confused_class_index[idx, j]
                    confusion_ratio = confusionmatrix_normlized[idx, idx2]
                    if confusion_ratio > 0.0:
                        logfp.write('  %s [%0.2f]' % (idx2, confusion_ratio))
                logfp.write('|  \n')

        data_flase = {
            'ids': [],
            'Label': [],
            'y_pred': [],
            'y_pred_score': [],
            'Cleaned_SMILES': []
        }
        logfp.write('___   \n')
        logfp.write('### **Error Details**   \n')
        logfp.write('Total %d out of %d samples mis-classified    \n' % (np.sum(err_num_per_class), test_sample_num))
        logfp.write('| Sample ID   |  Label  | Predicted Result  |  Score  |   Cleaned_SMILES   |\n')
        logfp.write('| ----------- | -------------- | ----------------  | ------- |  --------- |\n')
        for ID, SMILES, predict_result, groundtruth in zip(IDs_test, SMILES_test, predicted_results, groundtruths):  # TODO: set to original ID
            if int(predict_result[0]) != groundtruth:
                label_predict = int(predict_result[0])
                label_gt = groundtruth
                logfp.write(
                    '| %s | %d | %d | %0.4f | ```%s``` |\n' % (ID, label_gt, label_predict, predict_result[1], SMILES))
                data_flase['ids'].append(ID)
                data_flase['Label'].append(label_gt)
                data_flase['y_pred'].append(label_predict)
                data_flase['y_pred_score'].append(predict_result[1])
                data_flase['Cleaned_SMILES'].append(SMILES)
        # logfp.close()
        testlog.rpt_file = rpt_file
    df0 = pd.DataFrame(data_flase)
    df0.to_csv(pjoin(save_folder, result_prefix + 'pred_false.csv'))
    data_pred = {
            'ids': [],
            'y': [],
            'y_pred': [],
            'y_pred_score': [],
            'y_pred_0_score': scorematrix[:, 0],
            'y_pred_1_score': scorematrix[:, 1],
            'smiles': []
        }
    for ID, SMILES, predict_result, groundtruth in zip(IDs_test, SMILES_test, predicted_results,
                                                       groundtruths):  # TODO: set to original ID
        label_predict = int(predict_result[0])
        label_gt = groundtruth
        score = predict_result[1]
        if label_predict != groundtruth:
            label_gt = groundtruth
            logfp.write(
                '| %s | %d | %d | %0.4f | ```%s``` |\n' % (ID, label_gt, label_predict, predict_result[1], SMILES))
        data_pred['ids'].append(ID)
        data_pred['y'].append(label_gt)
        data_pred['y_pred'].append(label_predict)
        data_pred['y_pred_score'].append(score)
        data_pred['smiles'].append(SMILES)

    # Update: 2021/09/02
    df = pd.DataFrame(data_pred)
    df.to_csv(pjoin(save_folder, 'y_pred.csv'))

    return testlog


def model_predict(args):
    # --- public paras ---#
    device = args.device
    model_file = args.model_file
    model_ver = args.model_ver
    class_num = args.class_num
    pred_threshold = args.pred_threshold
    testset = args.testset
    batch_size = args.batchsize
    save_root_folder = args.save_root_folder
    use_multiprocessing = args.use_multiprocessing
    result_prefix = args.result_prefix
    config_set = args.config
    viz_att = args.viz_attention
    positive_only = args.positive_only
    test_loader_worker = 1  # fixed to 1
    testlog = TestLog()

    # --- (1) setup device ---#
    if device < 0:
        print('Using CPU for evaluation')
        device = torch.device('cpu')
    else:
        print('Using CUDA%d for evaluation' % device)
        device = torch.device('cuda:%d' % device)
        if viz_att and batch_size > 1:
         raise ValueError('To visualize the attention map, the batch size have to be 1')
    # --- (2) prepare model ---#
    if config_set is None:
        config_set = 'model_%s_config' % model_ver
    config = getattr(experiment_configs, config_set)()
    task = 'regression' if class_num == 1 else 'classification'
    model_agent = Model_Agent(device=device, model_ver=args.model_ver, output_dim=args.class_num, task=task,
                              config=config, model_file=model_file, atom_dict_file=args.atom_dict_file, viz_att=viz_att)
    if model_agent.model_file_md5 is not None:
        print('model weights loaded')
    else:
        raise ValueError('model file is not specified')

    print('local ip = %s, model ver = %s, time_stamp = %s' % (local_ip, model_ver, time_stamp))

    # --- (3) setup data ---#
    testset_data = gpickle.load(testset)
    if isinstance(testset_data, dict):
        mol_features_test = testset_data['features']
        SMILES_test = testset_data['SMILESs']
        IDs_test = testset_data['IDs']
    else:
        mol_features_test, _, SMILES_test, IDs_test, *aux_data_list = testset_data

    test_sample_num = SMILES_test.shape[0]
    if IDs_test is None:
        IDs_test = list(range(test_sample_num))
    if save_root_folder is None:
        save_root_folder = os.path.join(os.getcwd(), 'prediction')
    if not os.path.exists(save_root_folder):
        os.makedirs(save_root_folder)
    save_folder = os.path.join(save_root_folder, os.path.splitext(os.path.basename(model_file))[0])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    testlog.save_folder = save_folder
    queue_size = 3
    if use_multiprocessing:
        data_container = multiprocessing.Queue
    else:
        data_container = queue.Queue
    test_data_queue = data_container(queue_size * test_loader_worker)
    test_DLM = data_loader.Data_Loader_Manager(batch_data_loader=model_agent.batch_data_loader,
                                               data_queue=test_data_queue,
                                               data=(mol_features_test,),
                                               shuffle=False,
                                               batch_size=batch_size,
                                               worker_num=test_loader_worker,
                                               use_multiprocessing=use_multiprocessing,
                                               auto_rewind=0,
                                               name='test_DLM')
    # --- (4) evaluation ---#
    print('Predicting on test set (size = %d)' % test_sample_num)
    model_agent.model.train(False)
    predicted_results = []
    attention_map = []
    test_time0 = time.time()
    tested_sample_num = 0
    with torch.no_grad():
        while tested_sample_num < test_sample_num:
            time0 = time.time()
            batch_test_data = test_data_queue.get()
            time1 = time.time()
            scorematrix, att_map = model_agent.forward(batch_test_data)
            if positive_only:
                if class_num > 2:
                    raise ValueError('positive_only option is ONLY used for Binary Classification.')
            if task == 'classification':
                scorematrix = softmax(scorematrix, dim=1)
                score_mask_neg = torch.unsqueeze((scorematrix[:, 0] >= pred_threshold), 1)  #set threshold negitve
                score_mask_pos = torch.unsqueeze((scorematrix[:, 1] > (1 - pred_threshold)), 1)  #set threshold postive
                score_mask = torch.cat((score_mask_neg, score_mask_pos), dim =1)
                best_labels = torch.argmax(score_mask.long(), dim=1)
                best_ps = torch.masked_select(scorematrix, score_mask)
                # best_ps, best_labels = torch.max(scorematrix, dim=1)
                predicted_idxs = best_labels.cpu().detach().numpy().astype(np.int32)
                if positive_only:
                    best_ps = scorematrix[:, 1]
                predicted_scores = best_ps.cpu().detach().numpy()
                batch_sample_num = predicted_idxs.shape[0]
                for i in range(batch_sample_num):
                    predicted_results.append([predicted_scores[i], predicted_idxs[i]])
                    attention_map.append(att_map)
            else:
                predicted_scores = scorematrix.squeeze(1).cpu().detach().numpy()
                batch_sample_num = predicted_scores.shape[0]
                for i in range(batch_sample_num):
                    predicted_results.append([predicted_scores[i]])
                    attention_map.append(att_map)
            tested_sample_num += batch_sample_num
            time2 = time.time()
            data_process_time = time1 - time0
            test_time = time2 - time1
            progress = tested_sample_num / test_sample_num
            print(result_prefix + ': time = %0.2fs(%0.2f|%0.2f), progress = %0.2f%%, time remained = %0.2fmins' % (
                (time2 - time0), test_time, data_process_time, progress * 100,
                (time2 - test_time0) / 60 * (1 - progress) / progress))
    test_DLM.close()
    test_time2 = time.time()

    test_time_cost = test_time2 - test_time0

    print('test time cost = %0.2fmins, speed = %0.2f samples/s @batchsize = %d' % (
        test_time_cost / 60, tested_sample_num / test_time_cost, batch_size))
    for p in multiprocessing.active_children():
        p.terminate()
    if tested_sample_num != test_sample_num:
        raise RuntimeError('tested sample num = %d out of %d' % (tested_sample_num, test_sample_num))
        # print('tested sample num = %d out of %d' % (tested_sample_num, test_sample_num))

    # --- (4.1) merge & sort result ---#
    for idx, IDs_item in enumerate(IDs_test):
        predicted_results[idx].extend([SMILES_test[idx], IDs_item])
    if args.sort:
        scores = []
        for i in range(test_sample_num):
            scores.append(predicted_results[i][0])
        sorted_idxs = np.argsort(scores)
        sorted_idxs = sorted_idxs[::-1]
        sorted_results = []
        for i in range(test_sample_num):
            idx = sorted_idxs[i]
            sorted_results.append(predicted_results[idx])
    else:
        sorted_results = predicted_results
    result_file = os.path.join(save_folder, 'predictions%s.gpkl' % (time_stamp))  # TODO: change predictions format
    gpickle.dump(sorted_results, result_file)
    testlog.result_file = result_file

    # --- (5) create csv file ---#
    if positive_only:
        csv_file = os.path.join(save_folder, result_prefix + '%s_positive_predictions@%s.csv' % (
        task, time_stamp))  # active prediction
    else:
        csv_file = os.path.join(save_folder, result_prefix + '%s_predictions@%s.csv' % (task, time_stamp))
    sep = ','
    with open(csv_file, mode='wt', encoding='utf8') as f:
        if task == 'classification':
            if positive_only:
                log = sep.join(['ID', 'SMILES', 'Score'])
            else:
                log = sep.join(['ID', 'SMILES', 'Label', 'Score'])
        else:
            log = sep.join(['ID', 'SMILES', 'Score'])
        f.write('%s\n' % log)
        for i in range(test_sample_num):
            record = sorted_results[i]
            if task == 'classification':
                score, label_idx, SMILES, ID = record
                if positive_only:
                    log = sep.join(['%s' %ID, SMILES, '%0.4f' % score])
                else:
                    log = sep.join(['%s' %ID, SMILES, '%d' % label_idx, '%0.4f' % score])
            else:
                score, SMILES, ID = record
                log = sep.join([str(ID), SMILES, '%0.4f' % score])
            f.write('%s\n' % log)
    testlog.csv_file = csv_file
    if viz_att:
        for j in range(test_sample_num):
            record = sorted_results[j]
            if task == 'classification':
                score, label_idx, SMILES, ID = record
            elif task == 'regression':
                score, SMILES, ID = record
            if positive_only:
                visualize_atom_attention(save_folder,ID, SMILES, score, attention_map[j])
            else:
                visualize_atom_attention(save_folder,ID, SMILES, 1.0, attention_map[j])

    # --- (6) create report file ---#
    rpt_file = os.path.join(save_folder, result_prefix + 'predictions@%s.md' % (time_stamp))
    logfp = open(rpt_file, mode='wt', encoding='utf8')
    logfp.write('### **Parameters**  \n')
    logfp.write('| param                | value  |  \n')
    logfp.write('| -------------------  | -----: |  \n')
    logfp.write('| time stamp           |  %s    |  \n' % time_stamp)
    logfp.write('| local_ip             |  %s    |  \n' % local_ip)
    logfp.write('| device               |  %d    |  \n' % args.device)
    logfp.write('| model version        |  %s    |  \n' % model_ver)
    logfp.write('| model file           |  %s    |  \n' % model_file)
    logfp.write('| config               |  %s    |  \n' % config_set)
    logfp.write('| test_set             |  %s    |  \n' % testset)
    logfp.write('| test set size        |  %d    |  \n' % test_sample_num)
    logfp.write('| class_num            |  %d    |  \n' % class_num)
    logfp.write('| pred_threshold       |  %d    |  \n' % pred_threshold)
    logfp.write('| batch_size           |  %d    |  \n' % batch_size)
    logfp.write('| time cost            |  %0.2fmins |  \n' % (test_time_cost / 60))
    logfp.write('| samples/s            |  %0.2f |  \n' % (test_sample_num / test_time_cost))
    logfp.close()
    testlog.rpt_file = rpt_file

    return testlog


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-device', default=-1, type=int, help='device, -1=CPU, >=0=GPU')
    argparser.add_argument('-model_file', default=None, type=str)
    argparser.add_argument('-model_ver', default='4v4', type=str)
    argparser.add_argument('-pred_threshold', default=0.5, type=float,  help='set prediction threshold for NEGETIVE class, the positive class is automaticly set: (1- pred_threshold )')
    argparser.add_argument('-class_num', default=2, type=int, help='for regression task, set class_num to 1')
    argparser.add_argument('-testset', default='Mtb_mt_testset_3808_binary.gpkl', type=str)
    argparser.add_argument('-batchsize', default=1, type=int)
    argparser.add_argument('-save_root_folder', default=None, type=str)
    argparser.add_argument('-use_multiprocessing', default='true', type=str)
    argparser.add_argument('-result_prefix', default='', type=str)
    argparser.add_argument('-viz_attention', default=False, action='store_true', help='Whether to visualize attention map of graphs. ATTENTION: the batch size shoulde be 1')
    argparser.add_argument('-config', default=None, type=str, help='param configuration set')
    argparser.add_argument('-predict_only', action='store_true', help='only do prediction on given dataset')
    argparser.add_argument('-positive_only', action='store_true', help='only predict active probability score')
    argparser.add_argument('-sort', default='false', type=str,
                           help='whether sort result csv by predicted scores, default = True')
    argparser.add_argument('-load_weights_strict', default='true', type=str,
                           help='whether load model weights by strictly checking')
    args = argparser.parse_args()
    args.load_weights_strict = args.load_weights_strict.lower() in {'true', 'yes'}
    args.use_multiprocessing = args.use_multiprocessing.lower() in {'true', 'yes'}
    args.sort = args.sort.lower() in {'true', 'yes'}


    if args.model_ver in {'1'}:
        args.atom_dict_file = 'Mtb_mt_charset_canonicalized.gpkl'
    else:
        args.atom_dict_file = 'atom_dict.gpkl'

    torch.set_grad_enabled(False)
    if args.predict_only:
        testlog = model_predict(args=args)
    else:
        testlog = model_evaluate(args=args)
    print(testlog.__dict__)
    print('All done~')
