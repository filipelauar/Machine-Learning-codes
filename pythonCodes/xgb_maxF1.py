# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 14:12:45 2020

@author: Filipe
"""

# This code can be used as eval metric for the xgboost algorithm in python

# Input: Predictions probabilities (preds) and xgb.DMatrix (dmatrix)
# Output: Best F1-score for binary data

import numpy as np
from sklearn.metrics import precision_recall_curve

def xgb_maxF1(preds, dtrain):
    
    labels = dtrain.get_label()

    precision, recall, thresholds = precision_recall_curve(labels, preds)
    thresholds = np.append(thresholds, 1)
    f1_scores = 2*(precision*recall)/(precision+recall)

    best_row = f1_scores.argmax()

    if(best_row >= 0):
        return 'f1_score', f1_scores[best_row]
    else:
        return 'f1_score', -1