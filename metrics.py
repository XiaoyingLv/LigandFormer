# coding:utf-8
"""
Metric definitions
Created  :   7, 10, 2019
Revised  :   7, 10, 2019
Author   :  David Leon (dawei.leng@ghddi.org)
All rights reserved
-------------------------------------------------------------------------------
"""
__author__ = 'dawei.leng'

import scipy as sp
import sklearn.metrics as slmetrics
import numpy as np, warnings

def pearson_corr(y_score, y_true, sample_weight=None, drop_intermediate=True):
    """
    This `pearson_corr` function and the following `spearman_corr` function for regression problem.
    NOTE: this treatment is not accurate for multi-class problem, so the results are only for weak reference.
    :param y_true:
    :param y_score:
    :param sample_weight:
    :param drop_intermediate:
    :return:
    """
    pearson_value, p_value  =  sp.stats.pearsonr(y_score, y_true)
    return pearson_value, p_value

def spearman_corr(y_score, y_true, sample_weight=None, drop_intermediate=True):
    """
    This `spearman_corr` function and the  `pearson_corr`  function above for regression problem.
    NOTE: this treatment is not accurate for multi-class problem, so the results are only for weak reference.
    :param y_true:
    :param y_score:
    :param sample_weight:
    :param drop_intermediate:
    :return:
    """
    spearman_value, p_value  =  sp.stats.spearmanr(y_score, y_true)
    return spearman_value, p_value

def roc_curve(y_true, y_score, sample_weight=None, drop_intermediate=True):
    """
    This `roc_curve` function and the following `pr_curve` function treat multi-class problem as multi-label problem,
    and compute roc curve and au-roc for each label.
    NOTE: this treatment is not accurate for multi-class problem, so the results are only for weak reference.
    :param y_true:
    :param y_score:
    :param sample_weight:
    :param drop_intermediate:
    :return:
    """
    if len(y_score.shape) == 1:
        sample_num, class_num = y_score.shapep[0], 2
    else:
        sample_num, class_num = y_score.shape
    roc_curves = []
    for i in range(class_num):
        fpr, tpr, th = slmetrics.roc_curve(y_true, y_score[:,i], pos_label=i,
                                                 sample_weight=sample_weight,
                                                 drop_intermediate=drop_intermediate)
        auroc = slmetrics.auc(fpr, tpr)
        roc_curves.append([fpr, tpr, th, auroc])
    return roc_curves

def pr_curve(y_true, y_score, sample_weight=None):
    """
    This `pr_curve` function and the above `roc_curve` function treat multi-class problem as multi-label problem,
    and compute PR curve and au-pr for each label.
    NOTE: this treatment is not accurate for multi-class problem, so the results are only for weak reference.
    :param y_true:
    :param y_score:
    :param sample_weight:
    :return:
    """
    if len(y_score.shape) == 1:
        sample_num, class_num = y_score.shapep[0], 2
    else:
        sample_num, class_num = y_score.shape
    pr_curves = []
    for i in range(class_num):
        p, r, th = slmetrics.precision_recall_curve(y_true, y_score[:,i], pos_label=i,
                                                 sample_weight=sample_weight)
        aupr = slmetrics.auc(r, p)
        pr_curves.append([p, r, th, aupr])
    return pr_curves

def alpha_enrichment(y_true, y_score, alpha=None):
    """
    Compute the alpha-enrichment factor for model performance measuring, only for binary classification
    :param y_true: (batch_size,), int array with 1 for positive class and 0 for negative class
    :param y_score: score of the positive class, (batch_size,), float array with elements in range(0, 1.0)
    :param alpha: float in range(0, 1.0), should not be larger than the positive ratio in `y_ture`
    :return: alpha_acc, positive_ratio, relative_enrichment
             `alpha_acc` is the accuracy among the top `alpha` percent of the predictions
             `positive_ratio` is the positive class percent of ground truth
             `relative_enrichment` = `alpha_acc` / `positive_ratio`, represents the enrichment power of the model
             at the given top `alpha` percent of the model ranking results.
    """
    idxs = np.argsort(y_score)
    idxs = idxs[::-1]        # reverse
    n = len(y_true)
    p = (y_true>0).sum()
    if alpha is not None:
        m = max(1, int(alpha * n))
        if m > p:
            warnings.warn('alpha is too big, only %0.2f percent of y_true is positive' % (p/n*100))
            m = p
    else:
        m = p
    y_pred = y_score[idxs[:m]] > 0.5
    y_true = y_true[idxs[:m]]
    alpha_acc = y_true[y_pred].sum() / m
    positive_ratio = p / n
    relative_enrichment = alpha_acc / positive_ratio
    return alpha_acc, positive_ratio, relative_enrichment

def score_level_enrichment(y_true, y_score, score=0.5):
    """
    Compute the relative enrichment for di, only for binary classification
    :param y_true: (batch_size,), int array with 1 for positive class and 0 for negative class
    :param y_score: score of the positive class, (batch_size,), float array with elements in range(0, 1.0)
    :param score: float in range(0, 1.0), score thre
    :return: level_acc, positive_ratio, relative_enrichment
             `level_acc` is the accuracy among the predictions with score value > `score`
             `positive_ratio` is the positive class percent of ground truth
             `relative_enrichment` = `alpha_acc` / `positive_ratio`, represents the enrichment power of the model
             at the given top `alpha` percent of the model ranking results.
    """
    idxs = np.argsort(y_score)
    m = np.argmax(y_score[idxs] > score)
    n = len(y_true)
    p = (y_true>0).sum()
    y_pred = y_score[idxs[m:]] > 0.5
    y_true = y_true[idxs[m:]]
    level_acc = y_true[y_pred].sum() / (n-m)
    positive_ratio = p / n
    relative_enrichment = level_acc / positive_ratio
    return level_acc, positive_ratio, relative_enrichment

if __name__ == '__main__':
    if 0:
        from dandelion.util import gpickle
        from matplotlib import pyplot as plt

        gt, pred = gpickle.load("/Users/lengdawei/Work/Project/ligand_based_vs_models_deepchem/pytorch_version/evaluation/ligand_based_VS_model_1_pytorch@2019-07-11_09-17-05_N=868053_ER=[34.95, 18.30]/predictions.gpkl")
        # gt, pred = gpickle.load("/Users/lengdawei/Work/Project/ligand_based_vs_models_deepchem/pytorch_version/evaluation/ligand_based_VS_model_2_1_pytorch@2019-07-10_17-12-54_N=5665188_ER=[9.93, 9.38]/predictions.gpkl")

        print('gt.shape=', gt.shape)
        print('pred.shape=', pred.shape)

        roc_curves = roc_curve(gt, pred)
        for i, ftt in enumerate(roc_curves):
            fpr, tpr, th, auroc = ftt
            print('auroc[%d]=' % i, auroc)
            plt.plot(fpr, tpr)
        plt.legend(['roc[%d]' % i for i in range(len(roc_curves))])

        plt.figure()
        pr_curves = pr_curve(gt, pred)
        for i, prt in enumerate(pr_curves):
            p, r, th, aupr = prt
            print('aupr[%d]=' % i, aupr)
            plt.plot(r, p)
        plt.legend(['pr[%d]' % i for i in range(len(pr_curves))])

        plt.show()
    if 0:
        vp = verbose_print(0)
        vp('abc,')
        vp('123%s'%('abc'), end='\n\n', l=2)
