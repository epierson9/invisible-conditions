### This script treats the available toxicity label as y, simulates s,
### and compares PURPLE's performance to baselines.

import sys
import torch 
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, '../')
sys.path.insert(0, '../MIMIC_notebooks')
sys.path.insert(0, '../WILDS_notebooks')


from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split

from sarpu.pu_learning import *
from sarpu.PUmodels import LogisticRegressionPU
from km.Kernel_MPE_grad_threshold import wrapper as ramaswamy

from gpu_utils import restrict_GPU_pytorch
from method import train_relative_estimator, get_loss
from eval_fs import eval_relative_prior, eval_pred_prior, eval_PURPLE_relative_priors
from cmod_helper_fs import preprocess_cmod_data, normalize_x, MODE_GROUP_MAP
from baselines import cdmm, supervised_rel_prior, sar_em_rel_prior, scar_km2_rel_prior
from baseline_params import penalty, solver, fit_intercept
from cmod_helper_fs import get_group_weights

restrict_GPU_pytorch(sys.argv[4])

condition = sys.argv[1]
method = sys.argv[2]
mode = sys.argv[3]
groups = MODE_GROUP_MAP[mode]


if condition == 'cmod':
    split_dir = "../data/real_true_labels/content_mod/0.5/"
    results_f_name = "results/simulated_" + condition + "_" + method + "_" + mode + "_results"

# read features and labels
x = np.load(split_dir + 'vals.npz.npy')
y = np.load(split_dir + 'observed_labels.npy')
text = np.load(split_dir + 'text.npy')
toxicity = np.load(split_dir + 'toxicity.npy')

# meta experiment config
n_runs = 5
n_groups = len(groups)
n_attributes = x.shape[1]

scar_assumption = True
seeds = [10, 52, 30, 42, 1000]
classification_model_type = LogisticRegressionPU

labeling_frequencies = [.5 for group in groups]
labeling_frequencies[0] = .8

lamda_opts = [0, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
c2_opts = np.arange(.1, 1.0, .2)

classification_attributes = np.array(list(range(n_attributes))) + n_groups

### Enumerate experiment configurations

expmt_configs = []
for lamda in lamda_opts:
    for seed_i in range(n_runs):
        expmt_config = {'scar_assumption': scar_assumption, 
                        'lamda': lamda, 'n_groups': n_groups,
                        'n_attributes': n_attributes, 'groups': groups, 
                        'labeling_frequencies': tuple(labeling_frequencies), 
                        'category': mode, 'n_epochs': 10000,
                        'optimizer': 'Adam', 
                        'seed': seeds[seed_i], 
                        'estimator_type': 'logreg'} 
        expmt_configs.append(expmt_config)

### Run each experiment

result_dicts = []
for expmt_config in tqdm(expmt_configs):
    groups = expmt_config['groups']
    seed = expmt_config['seed']
    
    train, val, test = preprocess_cmod_data(x, y, s=[], toxicity=toxicity,
                                            text=text, groups=groups, expmt_config=expmt_config, random_seed=seed)
    
    x_train, y_train, s_train, _, _  = train
    x_val, y_val, s_val, _, _ = val
    x_test, y_test, s_test, _, _ = test
    
    x_train_norm, x_val_norm, x_test_norm = normalize_x(x_train, x_val, x_test, 
                                                        n_groups=expmt_config['n_groups'])
    
    expmt_config['group_weights'] = get_group_weights(x_train, expmt_config['n_groups'])

    g1_train_idxs = np.squeeze(np.array(x_train[:,0] == 1))
    g2_train_idxs = np.squeeze(np.array(x_train[:,1] == 1))
    g1_test_idxs = np.squeeze(np.array(x_test[:,0] == 1))
    g2_test_idxs = np.squeeze(np.array(x_test[:,1] == 1))

    info = {}           
    if method == 'ours':
        ## Train model
        f_model, losses, info = train_relative_estimator(x_train, s_train, 
                                                         x_val, s_val,
                                                         expmt_config,
                                                         save_model=True)

        ## Write out results for *all groups*
        result_dict_list = eval_PURPLE_relative_priors(x_test, f_model, groups)
        [result_dict.update(expmt_config) for result_dict in result_dict_list]
        [result_dict.update(info) for result_dict in result_dict_list]
        result_dicts.extend(result_dict_list)

    else:
        
        for i, group in enumerate(groups):
            classification_model1 = classification_model_type(fit_intercept=fit_intercept,
                                                             penalty=penalty, solver=solver)
            classification_model2 = classification_model_type(fit_intercept=fit_intercept,
                                                             penalty=penalty, solver=solver)
            g1_train_idxs = np.array(x_train[:,i] == 1).flatten()
            g2_train_idxs = np.array(x_train[:,i] == 0).flatten()
            g1_test_idxs = np.squeeze(np.array(x_test[:,i] == 1))
            g2_test_idxs = np.squeeze(np.array(x_test[:,i] == 0))

            
            if method == 'negative':
                results = supervised_rel_prior(x_train[g1_train_idxs], 
                                               x_train[g2_train_idxs],
                                               s_train[g1_train_idxs],
                                               s_train[g2_train_idxs], x_test, expmt_config,
                                               classification_model1=classification_model1,
                                               classification_model2=classification_model2)

                pred_rel_prior, pred_g1_prior, pred_g2_prior, _ = results

            elif method == 'sar-em':
                results = sar_em_rel_prior(x_train_norm[g1_train_idxs],
                                           x_train_norm[g2_train_idxs],
                                           s_train[g1_train_idxs],
                                           s_train[g2_train_idxs], x_test_norm, expmt_config,
                                               classification_model1=classification_model1,
                                               classification_model2=classification_model2)

                pred_rel_prior, pred_g1_prior, pred_g2_prior, _ = results


            elif method == 'scar-km2':
                results = scar_km2_rel_prior(x_train[g1_train_idxs], 
                                             x_train[g2_train_idxs],
                                             s_train[g1_train_idxs], 
                                             s_train[g2_train_idxs])
                pred_rel_prior, pred_g1_prior, pred_g2_prior, _ = results

            elif method == 'cdmm':
                pred_rel_prior, pred_g1_prior, pred_g2_prior = cdmm(x_train[g1_train_idxs],
                                                                    x_train[g2_train_idxs],
                                                                    s_train[g1_train_idxs],
                                                                    s_train[g2_train_idxs],
                                                                    expmt_config)

            elif method == 'observed':
                pred_g1_prior, pred_g2_prior = np.mean(s_train[g1_train_idxs]), np.mean(s_train[g2_train_idxs])
                pred_rel_prior = pred_g1_prior/pred_g2_prior

            elif method == 'true':
                pred_g1_prior, pred_g2_prior  = np.mean(y_test[g1_test_idxs]), np.mean(y_test[g2_test_idxs])
                pred_rel_prior = pred_g1_prior/pred_g2_prior

            result_dict = {'pred_rel_prior': pred_rel_prior, 
                           'pred_g1_prior': pred_g1_prior, 'pred_g2_prior': pred_g2_prior,
                           'method': method, 'group': group}
            result_dict.update(expmt_config)
            result_dict.update(info)
            result_dicts.append(result_dict)

    results_df = pd.DataFrame(result_dicts)
    results_df.to_csv(results_f_name)
