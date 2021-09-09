import os

import numpy as np
from sklearn import metrics
from sklearn.metrics import auc, precision_recall_curve


def aucs(l, s):
    fpr, tpr, thresholds = metrics.roc_curve(l, s, pos_label=1)
    AUC = auc(fpr, tpr)

    precision, recall, thresholds = precision_recall_curve(l, s)
    auc_precision_recall = auc(recall, precision)
    return AUC, auc_precision_recall


aucs(
    np.load(os.path.join('0905', 'l_front_d.npy')),
    np.load(os.path.join('0905', 's_front_d.npy')))

aucs(
    np.load(os.path.join('0905', 'l_front_ir.npy')),
    np.load(os.path.join('0905', 's_front_ir.npy')))

aucs(
    np.load(os.path.join('0905', 'l_front_ir.npy')),
    np.mean((np.load(os.path.join('0905', 's_front_ir.npy')),
             np.load(os.path.join('0905', 's_front_d.npy'))), axis=0))


aucs(
    np.load(os.path.join('0905', 'l_top_d.npy')),
    np.load(os.path.join('0905', 's_top_d.npy')))

aucs(
    np.load(os.path.join('0905', 'l_top_ir.npy')),
    np.load(os.path.join('0905', 's_top_ir.npy')))

aucs(
    np.load(os.path.join('0905', 'l_top_ir.npy')),
    np.mean((np.load(os.path.join('0905', 's_top_ir.npy')),
             np.load(os.path.join('0905', 's_top_d.npy'))), axis=0))


aucs(
    np.load(os.path.join('0905', 'l_top_ir.npy')),
    np.mean((np.load(os.path.join('0905', 's_front_d.npy')),
             np.load(os.path.join('0905', 's_top_d.npy'))), axis=0))

aucs(
    np.load(os.path.join('0905', 'l_top_ir.npy')),
    np.mean((np.load(os.path.join('0905', 's_front_ir.npy')),
             np.load(os.path.join('0905', 's_top_ir.npy'))), axis=0))

aucs(
    np.load(os.path.join('0905', 'l_top_ir.npy')),
    np.mean((np.load(os.path.join('0905', 's_front_ir.npy')),
             np.load(os.path.join('0905', 's_top_ir.npy')),
             np.load(os.path.join('0905', 's_front_d.npy')),
             np.load(os.path.join('0905', 's_top_d.npy'))), axis=0))
