# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 13:54:02 2020

@author: Filipe
"""

# Report the Threhsold, Recall, Precision and F1-score of the best F1-score based on the predicted probabilities

# Input: preds (model predicts), labels (true labels)
# Output: Threshold to get the best F1-score (prob), Recall of the best F1-score, Precision of the best F1-score, best F1-score

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

def maxF1_report(preds, labels):

    np.seterr(divide='ignore', invalid='ignore')
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    thresholds = np.append(thresholds, 1)
    f1_scores = 2*(precision*recall)/(precision+recall)

    df = pd.DataFrame({'precision' : precision, 'recall' : recall, 'thresholds' : thresholds, 'f1_score':f1_scores})

    #best_row = df['f1_score'].idxmax()

    return df.loc[df['f1_score'] == df.f1_score.max(),:]