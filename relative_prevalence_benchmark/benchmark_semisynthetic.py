import sys
import torch 
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, '../')
sys.path.insert(0, '../MIMIC_notebooks')
import pdb

from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split

from baselines import cdmm
from sarpu.pu_learning import *
from sarpu.PUmodels import LogisticRegressionPU
from km.Kernel_MPE_grad_threshold import wrapper as ramaswamy

from gpu_utils import restrict_GPU_pytorch
from simulation_helpers import generate_s_scar
from method import train_relative_estimator, get_loss
from eval_fs import eval_PURPLE_relative_priors
from mimic_helper_fs import get_idxs_of_group, preprocess_mimic_data, normalize_x, get_group_weights
from baselines import cdmm, supervised_rel_prior, sar_em_rel_prior, scar_km2_rel_prior
from baseline_params import penalty, solver, fit_intercept
from paths import RESULTS_DIR
print(penalty, solver, fit_intercept)

expmt_mode = sys.argv[1]
method = sys.argv[2]
restrict_GPU_pytorch(sys.argv[3])

if expmt_mode == 'random': 
    split_dir = "../data/semisynthetic/random/1/"
    results_f_name = RESULTS_DIR + "/random_" + method + "_semisimulated_results"
elif expmt_mode == 'ipv':
    split_dir = "../data/semisynthetic/ipv/1/"
    results_f_name = RESULTS_DIR + "/ipv_" + method + "_semisimulated_results"
elif expmt_mode == 'corr':
    split_dir = "../data/semisynthetic/corr/endometriosis/0/"
    results_f_name = RESULTS_DIR + "/corr4_" + method + "_semisimulated_results"
elif expmt_mode == 'high_rp':
    split_dir = "../data/semisynthetic/high_rp/0/"
    results_f_name = RESULTS_DIR + "/high_rp_" + method + "_semisimulated_results_.7"
    
# read features and labels
x = load_npz(split_dir + 'vals.npz').todense()
y = np.expand_dims(np.loadtxt(split_dir + 'positive_labels'), 1)
hadm_ids = np.loadtxt(split_dir + 'row_names')
subject_ids = np.loadtxt(split_dir + 'subject_ids')

# meta experiment config
n_runs = 5
n_epochs = 15000
n_attributes = x.shape[1]

c1 = .5 
c2_default = .5
scar_assumption = True
seeds = [10, 52, 30, 42, 1000]
classification_model_type = LogisticRegressionPU
groups = ['BLACK/AFRICAN AMERICAN', 'WHITE']

c2_opts = np.arange(.1, 1.0, .2)
c2_opts = [.9, .7, .5, .3]
c2_opts = [.7]
lamda_opts = [0, .01, .001, .0001, .00001]
lamda_opts = [.00001]
lamda_opts = [.01]

norm_mode = 'unnorm'
if method == 'sar-em':
    norm_mode = 'norm'

### Enumerate experiment configurations

expmt_configs = []
for  c2 in c2_opts:
    for lamda in lamda_opts:
        for seed in seeds[:n_runs]: 
            expmt_config = {'scar_assumption': scar_assumption, 
                            'labeling_frequency_g1': c1, 'labeling_frequency_g2': c2, 
                            'lamda': lamda, 'n_groups': len(groups),
                            'n_attributes': n_attributes, 'category': 'ethnicity',
                            'n_epochs': n_epochs, 'seed': seed, 'stratify_mode': 'none',
                            'estimator_type': 'logreg', 'mode': norm_mode,
                            'optimizer': 'Adam', 'group1_idx': 0} 
            expmt_configs.append(expmt_config)

g1_hadm_ids_idxs = get_idxs_of_group(hadm_ids, groups[0], 'ethnicity')
g2_hadm_ids_idxs = get_idxs_of_group(hadm_ids, groups[1], 'ethnicity')

### Subsample to 20% of the data for random and corr experiments

if expmt_mode in ['random', 'corr', 'high_rp']:
    subsample_pct = .2
    g1_hadm_ids_idxs = np.random.choice(g1_hadm_ids_idxs, 
                                        int(subsample_pct*len(g1_hadm_ids_idxs)), replace=False)
    g2_hadm_ids_idxs = np.random.choice(g2_hadm_ids_idxs, 
                                        int(subsample_pct*len(g2_hadm_ids_idxs)), replace=False)
    

g1_hadm_ids = hadm_ids[g1_hadm_ids_idxs]
g2_hadm_ids = hadm_ids[g2_hadm_ids_idxs]
g1_subject_ids = subject_ids[g1_hadm_ids_idxs]
g2_subject_ids = subject_ids[g2_hadm_ids_idxs]
x1, y1 = x[g1_hadm_ids_idxs], y[g1_hadm_ids_idxs]
x2, y2 = x[g2_hadm_ids_idxs], y[g2_hadm_ids_idxs] 

x = np.concatenate([x1, x2])
y = np.concatenate([y1, y2])
hadm_ids = np.concatenate([g1_hadm_ids, g2_hadm_ids])
subject_ids = np.concatenate([g1_subject_ids, g2_subject_ids])

### Run each experiment

result_dicts = []

for expmt_config in tqdm(expmt_configs):
    seed = expmt_config['seed']
    s1 = generate_s_scar(y1, expmt_config['labeling_frequency_g1'])
    s2 = generate_s_scar(y2, expmt_config['labeling_frequency_g2'])
    s = np.concatenate([s1, s2])
    
    print("METHOD: ", method)
    if method != 'ours' and expmt_config['lamda'] != .01:
        continue

    train, val, test = preprocess_mimic_data(x, y, s.flatten(), hadm_ids, 
                                 subject_ids, groups, expmt_config, random_seed=seed)

    x_train, y_train, s_train = train
    x_val, y_val, s_val = val
    x_test, y_test, s_test = test

    x_train_norm, x_val_norm, x_test_norm = normalize_x(x_train, x_val, x_test, 
                                                        n_groups=expmt_config['n_groups'])
    
    expmt_config['group_weights'] = get_group_weights(x_train, expmt_config['n_groups'])

    g1_train_idxs = np.squeeze(np.array(x_train[:,0] == 1))
    g2_train_idxs = np.squeeze(np.array(x_train[:,1] == 1))
    g1_test_idxs = np.squeeze(np.array(x_test[:,0] == 1))
    g2_test_idxs = np.squeeze(np.array(x_test[:,1] == 1))

    train_data, val_data, test_data = x_train, x_val, x_test
    if expmt_config['mode'] == 'norm':
        train_data, val_data, test_data = x_train_norm, x_val_norm, x_test_norm

    info, result_dict = {}, {}
    true_g1_prior = y_test[g1_test_idxs].mean()
    true_g2_prior = y_test[g2_test_idxs].mean()
    true_rel_prior = true_g1_prior / true_g2_prior
    true_result_dict = {'true_g1_prior': true_g1_prior, 'true_g2_prior': true_g2_prior,
                       'true_rel_prior': true_rel_prior}
    
    if method == 'ours':
        f_model, losses, info = train_relative_estimator(train_data, s_train, 
                                                         val_data, s_val,
                                                         expmt_config, save_model=True)
        result_dict_list = eval_PURPLE_relative_priors(test_data, f_model, groups)
        [result_dict.update(expmt_config) for result_dict in result_dict_list]
        [result_dict.update(info) for result_dict in result_dict_list]
        [result_dict.update(true_result_dict) for result_dict in result_dict_list]
        result_dicts.extend(result_dict_list)

    else:
        classification_model1 = classification_model_type(fit_intercept=fit_intercept,
                                                         penalty=penalty, solver=solver)
        
        classification_model2 = classification_model_type(fit_intercept=fit_intercept,
                                                         penalty=penalty, solver=solver)
        if method == 'supervised':
            results = supervised_rel_prior(x_train[g1_train_idxs], 
                                           x_train[g2_train_idxs],
                                           y_train[g1_train_idxs], 
                                           y_train[g2_train_idxs], x_test, expmt_config,
                                           classification_model1=classification_model1,
                                           classification_model2=classification_model2)

            pred_rel_prior, pred_g1_prior, pred_g2_prior, _ = results

        elif method == 'negative':
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
        elif method == 'true':
            pred_g1_prior = y1_test.mean()
            pred_g2_prior = y2_test.mean()
            pred_rel_prior = pred_g1_prior / pred_g2_prior
        
        result_dict.update(true_result_dict)
        result_dict.update({'pred_rel_prior': pred_rel_prior, 
                            'pred_g1_prior': pred_g1_prior, 'pred_g2_prior': pred_g2_prior,
                            'method': method, 'group': groups[0], 'seed': seed})
        
  
    result_dict.update(expmt_config)
    result_dict.update(info)
    result_dicts.append(result_dict)

    results_df = pd.DataFrame(result_dicts)
    results_df.to_csv(results_f_name)
    print(results_f_name)
