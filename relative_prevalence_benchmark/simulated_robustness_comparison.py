import numpy as np
import pandas as pd

import pdb
import sys
import torch

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from itertools import product

from simulation_helpers import separable_decision_rule_1d, inseparable_decision_rule_1d
from simulation_helpers import separable_decision_rule_nd, inseparable_decision_rule_nd
from simulation_helpers import create_gap_1d, create_gap_nd
from simulation_helpers import generate_g, generate_y, generate_s_scar
from sarpu.PUmodels import LogisticRegressionPU
from km.Kernel_MPE_grad_threshold import wrapper as ramaswamy

from sarpu.pu_learning import *
from baselines import cdmm, supervised_rel_prior, sar_em_rel_prior, scar_km2_rel_prior
from method import train_relative_estimator, get_loss
from eval_fs import eval_relative_prior, eval_pred_prior
from gpu_utils import restrict_GPU_pytorch
restrict_GPU_pytorch(sys.argv[2])
import itertools
np.random.seed(42)

# Fixed experimental parameters
method = sys.argv[1]
n_runs = 5
n_groups = 2
seeds = [10, 52, 30, 42, 1000, 1, 2, 3, 4, 29, 82374, 9]
scar_assumption = True
classification_model_type = LogisticRegressionPU
classification_model = classification_model_type(fit_intercept=True)

results_f_name = 'results/robustness_' + method + '_underestimate_results'

c1 = .5
c2_opts = [.3]
separability_opts =  ['overlap']
group_gap_opts = [-2, -1, -.5, -.25, 0]
p_y_diff_opts = [0, .1, .2, .3]
lamda_opts = [0]
std, n_to_sample_g1, n_to_sample_g2, n_attributes = 16, 10000, 10000, 5


# Specifying which set of experiments to run

# Enumerating the experiment configurations
# Relevant experiment params = c1, c2, n_dims, std, n_examples, group_gap
# TODO: Add iterations over gap between decision rules?
# call it decision_rule_diff

all_combinations = list(product(c2_opts, separability_opts, group_gap_opts, p_y_diff_opts))

# Creating group configurations based on each experiment config
expmt_configs = []
for lamda in lamda_opts:
    for  c2,  separability_assumption, group_gap, p_y_diff in all_combinations:
        # configs for generating group features
        g1_mean = np.ones(n_attributes)
        g2_mean = g1_mean + group_gap

        g1_config = {'mean': g1_mean, 'std': std, 'n_samples': n_to_sample_g1, 
                     'n_attributes': n_attributes, 'n_groups': n_groups, 'group_feat': 1}
        g2_config = {'mean': g2_mean, 'std': std, 'n_samples': n_to_sample_g2, 
                     'n_attributes': n_attributes, 'n_groups': n_groups, 'group_feat': 2}

        expmt_config = {'scar_assumption': scar_assumption, 
                        'separability_assumption': separability_assumption, 
                        'labeling_frequency_g1': c1, 'labeling_frequency_g2': c2, 
                        'group_gap': group_gap, 'lamda': lamda,
                       'n_groups': n_groups, 'n_attributes': n_attributes,
                       'p_y_diff': p_y_diff} 
        expmt_configs.append((expmt_config, g1_config, g2_config))
# Iterating over all experiments
result_dicts = []
for expmt_config, g1_config, g2_config in tqdm(expmt_configs):
    # generate features
    x1 = generate_g(g1_config)
    x2 = generate_g(g2_config)
         
    # generate true labels
    # TODO: change to accommodate different decision rules for each group
   
    beta = 1
    y1 = generate_y(x1, g1_config, inseparable_decision_rule_nd, beta,
                    decision_rule_diff=expmt_config['p_y_diff'])
    y2 = generate_y(x2, g2_config, inseparable_decision_rule_nd, beta)
    # generate observed labels
    if expmt_config['scar_assumption']:
        s1 = generate_s_scar(y1, expmt_config['labeling_frequency_g1'])
        s2 = generate_s_scar(y2, expmt_config['labeling_frequency_g2'])

    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    s = np.concatenate([s1, s2])
    
    for run in range(n_runs):
        x1_train, x1_test, y1_train, y1_test, s1_train, s1_test = train_test_split(x1, y1, s1,  test_size=.4,random_state=seeds[run],shuffle=True, stratify=y1)
        x2_train, x2_test, y2_train, y2_test, s2_train, s2_test = train_test_split(x2, y2, s2, test_size=.4, random_state=seeds[run], shuffle=True, stratify=y2)

        x1_val, x1_test, y1_val, y1_test, s1_val, s1_test = train_test_split(x1_test, y1_test, s1_test,  test_size=.5, shuffle=True, stratify=y1_test)
        x2_val, x2_test, y2_val, y2_test, s2_val, s2_test = train_test_split(x2_test, y2_test, s2_test, test_size=.5, shuffle=True, stratify=y2_test)

        x_train = np.concatenate([x1_train, x2_train])
        y_train = np.concatenate([y1_train, y2_train])
        s_train = np.concatenate([s1_train, s2_train])

        x_test = np.concatenate([x1_test, x2_test])
        y_test = np.concatenate([y1_test, y2_test])
        s_test = np.concatenate([s2_test, s2_test])
        
        x_val = np.concatenate([x1_val, x2_val])
        s_val = np.concatenate([s1_val, s2_val])
        
        scale_factor = np.max(np.abs(x_train))
        x_train_norm = x_train/scale_factor
        x_val_norm = x_val/scale_factor
        x_test_norm = x_test/scale_factor
    
        g1_train_idxs = x_train[:,0] == 1
        g2_train_idxs = x_train[:,1] == 1
        g1_test_idxs = x_test[:,0] == 1
        g2_test_idxs = x_test[:,1] == 1
        
        info = {}
        if method == 'ours':
            f_model, losses = train_relative_estimator(x_train, s_train, 
                                                       expmt_config, n_epochs=10000)

            pred_rel_prior, pred_g1_prior, pred_g2_prior = eval_relative_prior(x_test, f_model)

            s_preds = torch.squeeze(f_model(torch.Tensor(x_val).cuda()))
            val_loss = get_loss(s_val, s_preds)
            s_preds = torch.squeeze(s_preds).detach().cpu()
            auc = roc_auc_score(s_val, s_preds)
            auprc = average_precision_score(s_val, s_preds)
            info = {'auprc': auprc, 'auc': auc, 'val_loss': val_loss}

        elif method == 'supervised':
            results = supervised_rel_prior(x_train[g1_train_idxs], 
                                           x_train[g2_train_idxs],
                                           y1_train, y2_train, x_test, g1_config,
                                           classification_model=classification_model)

            pred_rel_prior, pred_g1_prior, pred_g2_prior = results

        elif method == 'negative':
            results = supervised_rel_prior(x_train[g1_train_idxs], 
                                           x_train[g2_train_idxs],
                                           s_train[g1_train_idxs],
                                           s_train[g2_train_idxs], x_test, g1_config,
                                           classification_model=classification_model)

            pred_rel_prior, pred_g1_prior, pred_g2_prior = results

        elif method == 'sar-em':
            results = sar_em_rel_prior(x_train_norm[g1_train_idxs],
                                       x_train_norm[g2_train_idxs],
                                       s_train[g1_train_idxs],
                                       s_train[g2_train_idxs], x_test_norm, g1_config,
                                       classification_model=classification_model)

            pred_rel_prior, pred_g1_prior, pred_g2_prior = results

        elif method == 'cdmm':
            results  = cdmm(x_train_norm[g1_train_idxs],
                            x_train_norm[g2_train_idxs],
                            s_train[g1_train_idxs], s_train[g2_train_idxs], g1_config)

            pred_rel_prior, pred_g1_prior, pred_g2_prior = results
            
        elif method == 'scar-km2':
            results = scar_km2_rel_prior(x1_train, x2_train, s1_train, s2_train)
            
            pred_rel_prior, pred_g1_prior, pred_g2_prior = results

        true_g1_prior = y_test[g1_test_idxs].mean()
        true_g2_prior = y_test[g2_test_idxs].mean()
        true_rel_prior = true_g1_prior / true_g2_prior

        result_dict = {'pred_rel_prior': pred_rel_prior, 'true_rel_prior': true_rel_prior, 
                       'rel_prior_err': true_rel_prior - pred_rel_prior,
                       'pred_g1_prior': pred_g1_prior, 'pred_g2_prior': pred_g2_prior,
                       'true_g1_prior': true_g1_prior, 'true_g2_prior': true_g2_prior,
                       'method': method, 'run': run}
        result_dict.update(**{"g1_" + k: v for k, v in g1_config.items()})
        result_dict.update(**{"g2_" + k: v for k, v in g2_config.items()})
        result_dict.update(expmt_config)
        result_dict.update(info)
        result_dicts.append(result_dict)

    results_df = pd.DataFrame(result_dicts)
    results_df.to_csv(results_f_name)