# This code was inspired by Laurae github repository
# Link: https://github.com/Laurae2/Laurae

# This code can be used as eval metric for the xgboost algorithm in R

# Input: Predictions probabilities (preds) and xgb.DMatrix (dmatrix)
# Output: Best F1-score for binary data


maxF1 <- function(preds, dmatrix) 
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
  DT[, f1s := 2 * precision * recall / (precision + recall)]
  
  best_row <- which.max(DT$f1s)
  
  if (length(best_row) > 0) {
    return(list(metric = "f1s", value = DT$f1s[best_row[1]]))
  } else {
    return(list(metric = "f1s", value = -1))
  }
}
