import sys
import torch 
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import pdb
sys.path.insert(0, '../')
sys.path.insert(0, '../MIMIC_notebooks')


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

from mimic_helper_fs import MODE_GROUP_MAP, preprocess_mimic_data, normalize_x, get_group_weights

condition = sys.argv[1]
method = sys.argv[2]
mode = sys.argv[3]
restrict_GPU_pytorch(sys.argv[4])
stratify_mode = 'none'
if len(sys.argv) > 5:
    stratify_mode = sys.argv[5]
groups = MODE_GROUP_MAP[mode]

split_dir = "../data/real_true_labels/hospital/ipv/0/"
results_f_name = "results/real_" + condition + "_" + method + "_" + mode + "_results_test"
if stratify_mode != 'dummy':
    results_f_name += 'stratified_by_' + stratify_mode

# read features and labels
x = load_npz(split_dir + 'vals.npz').todense()
s = np.loadtxt(split_dir + "observed_labels")
hadm_ids = np.loadtxt(split_dir + 'row_names')
subject_ids = np.loadtxt(split_dir + 'subject_ids')

# meta experiment config
n_runs = 5
n_groups = len(groups)
n_attributes = x.shape[1]

scar_assumption = True
seeds = [10, 52, 30, 42, 1000]
classification_model_type = LogisticRegressionPU

lamda_opts = [0, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
lamda_opts = [0, 1e-6, 1e-5, 1e-4]
classification_attributes = np.array(list(range(n_attributes))) + n_groups

norm_flag = 'unnorm'
if method in ['sar-em', 'negative']:
    norm_flag = 'norm'

### Enumerate experiment configurations
expmt_configs = []
for lamda in lamda_opts:
    for i in range(n_runs):
        expmt_config = {'scar_assumption': True, 
                        'lamda': lamda, 'n_groups': n_groups,
                        'n_attributes': n_attributes, 'groups': groups, 
                        'labeling_frequencies': (None, None),
                        'category': mode, 'n_epochs': 10000, 
                        'optimizer': 'Adam',
                        'seed': seeds[i], 
                        'estimator_type': 'logreg', 
                        'mode': norm_flag, 'n_batches': 20,
                        'stratify_mode': stratify_mode} 
        expmt_configs.append(expmt_config)

### Run each experiment
result_dicts = []
for expmt_config in tqdm(expmt_configs):
    groups = expmt_config['groups']
    seed = expmt_config['seed']
    
    ### Preprocess mimic data 
    train, val, test = preprocess_mimic_data(x, [], s, hadm_ids, subject_ids,
                                             groups, expmt_config, seed)
    
    x_train, y_train, s_train = train
    x_val, y_val, s_val = val
    x_test, y_test, s_test = test
    
    x_train_norm, x_val_norm, x_test_norm = normalize_x(x_train, x_val, x_test,
                                                        n_groups=expmt_config['n_groups'])
    
    train_data, val_data, test_data = x_train, x_val, x_test
    if expmt_config['mode'] == 'norm':
        train_data, val_data, test_data = x_train_norm, x_val_norm, x_test_norm

    expmt_config['group_weights'] = get_group_weights(x_train, expmt_config['n_groups'])
    
    info = {}           
    if method == 'ours':
        ### Training
        f_model, losses, info = train_relative_estimator(train_data, s_train, val_data, s_val, expmt_config, save_model=True)

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
            g1_test_idxs = np.squeeze(np.array(x_test[:,i] == 1))
            g2_test_idxs = np.squeeze(np.array(x_test[:,i] == 0))  
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
                results = sar_em_rel_prior(train_data[g1_train_idxs], train_data[g2_train_idxs],
                                           s_train[g1_train_idxs], s_train[g2_train_idxs], 
                                           test_data, expmt_config,
                                           classification_model1=classification_model1,
                                           classification_model2=classification_model2)

                pred_rel_prior, pred_g1_prior, pred_g2_prior, _ = results


            elif method == 'scar-km2':
                results = scar_km2_rel_prior(train_data[g1_train_idxs], train_data[g2_train_idxs], 
                                             s_train[g1_train_idxs], s_train[g2_train_idxs])
                
                pred_rel_prior, pred_g1_prior, pred_g2_prior, _ = results

            elif method == 'cdmm':
                results = cdmm(train_data[g1_train_idxs], train_data[g2_train_idxs],
                               s_train[g1_train_idxs], s_train[g2_train_idxs], expmt_config)

                pred_rel_prior, pred_g1_prior, pred_g2_prior = results
                
            elif method == 'observed':
                pred_g1_prior = np.mean(s_train[g1_train_idxs])
                pred_g2_prior = np.mean(s_train[g2_train_idxs])
                pred_rel_prior = pred_g1_prior/pred_g2_prior

            result_dict = {'pred_rel_prior': pred_rel_prior, 
                           'pred_g1_prior': pred_g1_prior, 'pred_g2_prior': pred_g2_prior,
                           'method': method, 'seed': seed, 'group': group}
            
            result_dict.update(expmt_config)
            result_dict.update(info)
            result_dicts.append(result_dict)

    results_df = pd.DataFrame(result_dicts)
    results_df.to_csv(results_f_name)
