# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 13:54:04 2020

@author: Filipe
"""


# Report the Threhsold and Balanced Accuracy of the best Balanced Accuracy based on the predicted probabilities

# Input: preds (model predicts), labels (true labels)
# Output: Threshold to get the best Balanced Accuracy (prob), best Balanced Accuracy


from sklearn.utils import column_or_1d
from sklearn.utils.extmath import stable_cumsum
import numpy as np
import pandas as pd
np.seterr(invalid='ignore')

def balanced_accuracy_report(preds, labels):
    
    y_true=labels
    y_score=preds
    
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    
    y_true = (y_true == 1)
    
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    num_labels = y_score.size
    
    fns = tps[-1] - tps
    tns = num_labels - (tps + fps + fns)
    
    sensitivity = tps / (tps + fps)
    specitivity = tns / (tns + fns)
    
    balanced_acc = (sensitivity + specitivity)/2
    balanced_acc[np.isnan(balanced_acc)] = 0
    
    threhsold = y_score[threshold_idxs]
    
    df = pd.DataFrame({'threhsolds' : threhsold, 'balanced_acc' : balanced_acc})
    
    return df.loc[df['balanced_acc'] == df.balanced_acc.max(),:]

