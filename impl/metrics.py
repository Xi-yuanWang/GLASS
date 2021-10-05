from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import numpy as np
from ogb.linkproppred import Evaluator


def binaryf1(pred, label):
    '''
    pred, label are numpy array
    can process multi-label target
    '''
    pred_i = (pred > 0).astype(np.int64)
    label_i = label.reshape(pred.shape[0],-1)
    return f1_score(label_i, pred_i, average="micro")


def microf1(pred, label):
    '''
    multi-class
    '''
    pred_i = np.argmax(pred, axis=1)
    # print(confusion_matrix(label,pred_i), flush=True)
    return f1_score(label, pred_i, average="micro")


def auroc(pred, label):
    return roc_auc_score(label, pred)

