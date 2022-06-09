import pdb
import torch
import numpy as np

def eval_PURPLE_relative_priors(x, f_model, groups):
    n_groups = len(groups)
    result_dict_list = []
    for i, group in enumerate(groups):
        g1_idxs = np.array(x[:,i] != 0).flatten()
        g2_idxs = np.array(x[:,i] == 0).flatten()

        x1_test_attributes = torch.Tensor(x[g1_idxs,n_groups:]).cuda()
        x2_test_attributes = torch.Tensor(x[g2_idxs,n_groups:]).cuda()
        results =  f_model.estimate_ratio(x1_test_attributes, x2_test_attributes)
        pred_rel_prior, pred_g1_prior, pred_g2_prior = results

        result_dict = {'pred_rel_prior': pred_rel_prior, 
                       'pred_g1_prior': pred_g1_prior,
                       'pred_g2_prior': pred_g2_prior, 'group': group, 'method': 'ours'}
        result_dict_list.append(result_dict)
    return result_dict_list
    
def eval_relative_prior(x, f1_model, f2_model, group1_idx, n_groups):
    g2_grp_idxs = [i for i in range(n_groups) if i != group1_idx]
    g1_idxs = np.array(x[:,group1_idx] != 0).flatten()
    g2_idxs = ~g1_idxs

    y1_pred = f1_model.predict_proba(x[g1_idxs])
    y2_pred = f2_model.predict_proba(x[g2_idxs])
    pred_g1_prior = y1_pred.mean()
    pred_g2_prior = y2_pred.mean()
    pred_rel_prior = pred_g1_prior / pred_g2_prior
    
    return pred_rel_prior, pred_g1_prior, pred_g2_prior

def eval_pred_prior(x, f_model):
    y_pred = f_model.predict_proba(x)
    return y_pred.mean()