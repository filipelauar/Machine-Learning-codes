# This code was inspired by Laurae github repository
# Link: https://github.com/Laurae2/Laurae

# Report the Threhsold, Recall, Precision and F1-score of the best F1-score based on the predicted probabilities
# and respecting the Precision and Recall conditions.

# Input: Predictions probabilities (preds), xgb.DMatrix (dmatrix), minimal accepted precision (min_precision), minimal accepted recall (min_recall)
# Output: Threshold to get the best F1-score (prob), Recall of the best F1-score, Precision of the best F1-score, best F1-score


xgb_maxF1_report <- function(preds, dmatrix, min_precision = 0., min_recall = 0.) 
{
  
  y_true <- getinfo(dmatrix, "label")
  
  DT <- data.table(y_true = y_true, y_prob = preds, key = "y_prob")
  cleaner <- !duplicated(DT[, "y_prob"], fromLast = TRUE)
  nump <- sum(y_true)
  numn <- length(y_true) - nump
  
  DT[, fp_v := cumsum(y_true == 1)]
  DT[, tp_v := nump - fp_v]
  DT[, fn_v := numn - as.numeric(cumsum(y_true == 0))]
  DT <- DT[cleaner, ]
  DT[, recall := (tp_v / (tp_v + fp_v))]
  DT[, precision := (tp_v / (tp_v + fn_v))]
  DT[, f1s := 2 * precision * recall^2 / (precision + recall)]
  
  DT <- DT[which(DT$precision >= min_precision & DT$recall >= min_recall)]
  best_row <- which.max(DT$f1s)
  
  if (length(best_row) > 0) {
    return(list(metric = "f1s", prob = DT$y_prob[best_row[1]], recall = DT$recall[best_row[1]], precision = DT$precision[best_row[1]], f1 = DT$f1s[best_row[1]]))
  } else {
    return(list(metric = "f1s", value = -1))
  }
}
