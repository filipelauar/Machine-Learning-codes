# Code translated from python to R from Imbalace-XGBoost repository
# Link: https://github.com/jhwjhw0123/Imbalance-XGBoost

# The loss was purposed in the paper: "Focal Loss for Dense Object Detection"
# Link: https://arxiv.org/pdf/1708.02002.pdf

# The gamma_indict is a trainable parameter


xgb_focal_loss <- function(pred, dmatrix)
{
  robust_pow <- function(num_base, num_pow)
  {
    return(sign(num_base) * (abs(num_base)) ** (num_pow))
  }
  
  #Include gamma value
  gamma_indct = 3
  
  # labels from xgboost dmatrix
  label = getinfo(dmatrix, "label")
  
  # compute the prediction with sigmoid
  sigmoid_pred = 1.0 / (1.0 + exp(-pred))
  
  # gradient
  # complex gradient with different parts
  g1 = sigmoid_pred * (1 - sigmoid_pred)
  g2 = label + ((-1) ** label) * sigmoid_pred
  g3 = sigmoid_pred + label - 1
  g4 = 1 - label - ((-1) ^ label) * sigmoid_pred
  g5 = label + ((-1) ^ label) * sigmoid_pred
  
  # combine the gradient
  grad = gamma_indct * g3 * robust_pow(g2, gamma_indct) * log(g4 + 1e-9) + ((-1) ** label) * robust_pow(g5, (gamma_indct + 1))
  
  # combine the gradient parts to get hessian components
  hess_1 = robust_pow(g2, gamma_indct) + gamma_indct * ((-1) ** label) * g3 * robust_pow(g2, (gamma_indct - 1))
  hess_2 = ((-1) ** label) * g3 * robust_pow(g2, gamma_indct) / g4
  
  # get the final 2nd order derivative
  hess = ((hess_1 * log(g4 + 1e-9) - hess_2) * gamma_indct +
            (gamma_indct + 1) * robust_pow(g5, gamma_indct)) * g1
  
  return(list(grad = grad, hess = hess))
}
