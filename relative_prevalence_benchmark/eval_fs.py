import pdb
import torch
import numpy as np

def eval_relative_prior(x, f1_model, f2_model=None):
    g1_idxs = np.array(x[:,0] != 0).flatten()
    g2_idxs = np.array(x[:,1] != 0).flatten()
        
    if f2_model:
        # Means we're approximating from relative methods
        y1_pred = f1_model.predict_proba(x[g1_idxs])
        y2_pred = f2_model.predict_proba(x[g2_idxs])
        pred_g1_prior = y1_pred.mean()
        pred_g2_prior = y2_pred.mean()
        pred_rel_prior = pred_g1_prior / pred_g2_prior
    else:
        # fix hard coding here
        x1_test_attributes = torch.Tensor(x[g1_idxs,2:]).cuda()
        x2_test_attributes = torch.Tensor(x[g2_idxs,2:]).cuda()
        pred_rel_prior, pred_g1_prior, pred_g2_prior =  f1_model.estimate_ratio(x1_test_attributes, x2_test_attributes)
    return pred_rel_prior, pred_g1_prior, pred_g2_prior

def eval_pred_prior(x, f_model):
    y_pred = f_model.predict_proba(x)
    return y_pred.mean()