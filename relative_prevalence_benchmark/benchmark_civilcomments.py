### This script treats the available toxicity label as s,
### and compares PURPLE's performance to baselines.

import pdb
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, '../')
sys.path.insert(0, '../WILDS_notebooks')

from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split

from sarpu.pu_learning import *
from sarpu.PUmodels import LogisticRegressionPU
from km.Kernel_MPE_grad_threshold import wrapper as ramaswamy

from gpu_utils import restrict_GPU_pytorch
from simulation_helpers import generate_s_scar
from method import train_relative_estimator, get_loss
from eval_fs import eval_relative_prior, eval_PURPLE_relative_priors
from baselines import cdmm, supervised_rel_prior, sar_em_rel_prior, scar_km2_rel_prior
from baseline_params import penalty, solver, fit_intercept

from cmod_helper_fs import preprocess_cmod_data, MODE_GROUP_MAP, normalize_x, MODE_GROUP_MAP_WITH_OTHER
from cmod_helper_fs import get_group_weights

condition = sys.argv[1]
method = sys.argv[2]
mode = sys.argv[3]
restrict_GPU_pytorch(sys.argv[4])

stratify_mode = 'none'
if len(sys.argv) > 5:
    stratify_mode = sys.argv[5]
    
groups = MODE_GROUP_MAP_WITH_OTHER[mode]

if condition == 'cmod':
    split_dir = "../data/real_true_labels/content_mod/0.5/"
    results_f_name = "results/real_" + condition + "_" + method + "_" + mode + "_results"
    
elif condition == 'cmod_higher_thresh':
    split_dir = "../data/real_true_labels/content_mod/0.6/"
    results_f_name = "results/real_" + condition + "_" + method + "_" + mode + "_results"

elif condition == 'cmod_highest_thresh':
    split_dir = "../data/real_true_labels/content_mod/0.8/"
    results_f_name = "results/real_" + condition + "_" + method + "_" + mode + "_results"

results_f_name += 'stratified_by_' + stratify_mode

# read features and labels
x = np.load(split_dir + 'vals.npz.npy')
s = np.load(split_dir + 'observed_labels.npy')
text = np.load(split_dir + 'text.npy')
toxicity = np.load(split_dir + 'toxicity.npy')

# meta experiment config
n_runs = 5
n_groups = len(groups)
n_attributes = x.shape[1]

scar_assumption = True
seeds = [10, 52, 30, 42, 1000]
classification_model_type = LogisticRegressionPU

c2_opts = np.arange(.1, 1.0, .2)
if method == 'ours':
    lamda_opts = [0, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
else:
    lamda_opts = [0]

## REMOVE 
lamda_opts = [1e-4]
classification_attributes = np.array(list(range(n_attributes))) + n_groups

norm_flag = 'unnorm'
if method in ['sar-em', 'negative']:
    norm_flag = 'norm'
    
### Enumerate experiment configurations

expmt_configs = []
for lamda in lamda_opts:
    for seed_i in range(n_runs):
        expmt_config = {'scar_assumption': scar_assumption, 
                        'lamda': lamda, 'n_groups': n_groups,
                        'n_attributes': n_attributes,'groups': groups,
                        'labeling_frequencies': (None, None),
                        'category': mode, 'n_epochs': 10000,
                        'optimizer': 'Adam', 
                        'seed': seeds[seed_i], 
                        'estimator_type': 'logreg', 'n_batches': 5,
                        'mode': norm_flag, 'stratify_mode': stratify_mode} 
        expmt_configs.append(expmt_config)

### Run each experiment

result_dicts = []
for expmt_config in tqdm(expmt_configs):
    groups = expmt_config['groups']
    seed = expmt_config['seed']
    
    train, val, test = preprocess_cmod_data(x, y=[], s=s, toxicity=toxicity, text=text,groups=groups, 
                                            expmt_config=expmt_config, random_seed=seed)
    
    x_train, y_train, s_train, _, _ = train
    x_val, y_val, s_val, _, _ = val
    x_test, y_test, s_test, _, _ = test
    
    x_train_norm, x_val_norm, x_test_norm = normalize_x(x_train, x_val, x_test,
                                                        n_groups=expmt_config['n_groups'])
    train_data, val_data, test_data = x_train, x_val, x_test
    if expmt_config['mode'] == 'norm':
        train_data, val_data, test_data = x_train_norm, x_val_norm, x_test_norm

    expmt_config['group_weights'] = get_group_weights(x_train, expmt_config['n_groups'])
    info = {}
    if method == 'ours':
        ### Training
        f_model, losses, info = train_relative_estimator(train_data, s_train, 
                                                         val_data, s_val,
                                                         expmt_config,
                                                         save_model=True)
        
        ### Evaluation 
        result_dict_list = eval_PURPLE_relative_priors(test_data, f_model, groups)
        [result_dict.update(expmt_config) for result_dict in result_dict_list]
        [result_dict.update(info) for result_dict in result_dict_list]
        result_dicts.extend(result_dict_list)

    else:
        
        for i, group in enumerate(groups):
            expmt_config['group1_idx'] = i
            expmt_config['group'] = group
            
            g1_train_idxs = np.array(train_data[:,i] == 1).flatten()
            g2_train_idxs = np.array(train_data[:,i] == 0).flatten()
            g1_test_idxs = np.squeeze(np.array(test_data[:,i] == 1))
            g2_test_idxs = np.squeeze(np.array(test_data[:,i] == 0))    
            classification_model1 = classification_model_type(fit_intercept=fit_intercept,
                                                             penalty=penalty, solver=solver)
            classification_model2 = classification_model_type(fit_intercept=fit_intercept,
                                                             penalty=penalty, solver=solver)

            
            if method == 'negative':
                results = supervised_rel_prior(train_data[g1_train_idxs], 
                                               train_data[g2_train_idxs],
                                               s_train[g1_train_idxs],
                                               s_train[g2_train_idxs], test_data, expmt_config,
                                               classification_model1=classification_model1,
                                               classification_model2=classification_model2)

                pred_rel_prior, pred_g1_prior, pred_g2_prior, _ = results

            elif method == 'sar-em':
                results = sar_em_rel_prior(train_data[g1_train_idxs],
                                           train_data[g2_train_idxs],
                                           s_train[g1_train_idxs],
                                           s_train[g2_train_idxs], test_data, expmt_config,
                                           classification_model1=classification_model1,
                                           classification_model2=classification_model2)

                pred_rel_prior, pred_g1_prior, pred_g2_prior, _ = results


            elif method == 'scar-km2':
                results = scar_km2_rel_prior(train_data[g1_train_idxs], 
                                             train_data[g2_train_idxs], 
                                             s1_train, s2_train)
                pred_rel_prior, pred_g1_prior, pred_g2_prior, _ = results

            elif method == 'cdmm':
                pred_rel_prior, pred_g1_prior, pred_g2_prior = cdmm(train_data[g1_train_idxs],
                                                                    train_data[g2_train_idxs],
                                                                    s_train[g1_train_idxs],
                                                                    s_train[g2_train_idxs],
                                                                    expmt_config)

            elif method == 'observed':
                pred_g1_prior = np.mean(s_train[g1_train_idxs])
                pred_g2_prior = np.mean(s_train[g2_train_idxs])
                pred_rel_prior = pred_g1_prior/pred_g2_prior

            result_dict = {'pred_rel_prior': pred_rel_prior, 
                           'pred_g1_prior': pred_g1_prior, 'pred_g2_prior': pred_g2_prior,
                           'method': method}
            result_dict.update(expmt_config)
            result_dict.update(info)
            result_dicts.append(result_dict)

    results_df = pd.DataFrame(result_dicts)
    results_df.to_csv(results_f_name)
