import sys
import torch
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

from sarpu.PUmodels import LogisticRegressionPU
from simulation_helpers import create_gap_1d, create_gap_nd
from simulation_helpers import generate_g, generate_y, generate_s_scar
from simulation_helpers import separable_decision_rule_1d, inseparable_decision_rule_1d
from simulation_helpers import separable_decision_rule_nd, inseparable_decision_rule_nd

from eval_fs import eval_relative_prior
from method import train_relative_estimator, get_loss
from baselines import cdmm, supervised_rel_prior, sar_em_rel_prior, scar_km2_rel_prior

from gpu_utils import restrict_GPU_pytorch
restrict_GPU_pytorch(sys.argv[3])
np.random.seed(42)

### Specifying default experiment parameters 


method = sys.argv[2]
methods = [method]
expmt_type = sys.argv[1]

n_runs = 5
n_groups = 2
scar_assumption = True
seeds = [10, 52, 30, 42, 1000]
classification_model_type = LogisticRegressionPU

c1 = .5
c2_default = .3
group_gap_default = -2
separability_default =  'overlap'
std, n_to_sample_g1, n_to_sample_g2, n_attributes = 16, 10000, 20000, 5

# Varying experimental parameters
c2_opts = [.1, .3, .5, .7, .9]
separability_opts = ['separable', 'overlap', 'overlaid']
group_gap_opts = [-2, -1, -.5, -.25, 0]
lamda_opts = [0]

expmt_type = sys.argv[1]
method = sys.argv[2]
methods = [method]

### Enumerating the experiment configurations

if expmt_type ==  'group_gap':
        all_combinations = [(c2_default, separability_default, group_gap_opt) for group_gap_opt in group_gap_opts]
        results_f_name = 'results/group_gap_' + method + '_results'
elif expmt_type == 'separability':
    all_combinations = [(c2_default, separability_opt, group_gap_default) for separability_opt in separability_opts]
    results_f_name = 'results/separability_' + method + '_results'
    if n_attributes == 1:
        results_f_name = 'results/separability_' + method + '_results_1d'

elif expmt_type == 'label_freq':
    all_combinations = [(c2_opt, separability_default, group_gap_default) for c2_opt in c2_opts]
    results_f_name = 'results/label_freq_' + method + '_results'

### Creating group configurations based on each experiment config

expmt_configs = []
for lamda in lamda_opts:
    for  c2,  separability_assumption, group_gap in all_combinations:
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
                       'n_groups': n_groups, 'n_attributes': n_attributes} 
        expmt_configs.append((expmt_config, g1_config, g2_config))

### Iterating over all experiments

result_dicts = []
for expmt_config, g1_config, g2_config in tqdm(expmt_configs):

    ### Generate the features for each group
    x1 = generate_g(g1_config)
    x2 = generate_g(g2_config)
         
    ### Generate true labels y for each group
    if expmt_config['separability_assumption'] == 'separable':
        y1 = generate_y(x1, g1_config, separable_decision_rule_nd)
        y2 = generate_y(x2, g2_config, separable_decision_rule_nd)
        x1, y1 = create_gap_nd(x1, y1, g1_config)
        x2, y2 = create_gap_nd(x2, y2, g2_config)
    elif expmt_config['separability_assumption'] == 'overlap' :
        beta = 1
        y1 = generate_y(x1, g1_config, inseparable_decision_rule_nd, beta)
        y2 = generate_y(x2, g2_config, inseparable_decision_rule_nd, beta)
    elif expmt_config['separability_assumption'] == 'overlaid':
        beta = 1
        y1 = generate_y(x1, g1_config, inseparable_decision_rule_nd, beta, scale=.75)
        y2 = generate_y(x2, g2_config, inseparable_decision_rule_nd, beta, scale=.75)
        
    ### Generate observed labels s for each group
    if expmt_config['scar_assumption']:
        s1 = generate_s_scar(y1, expmt_config['labeling_frequency_g1'])
        s2 = generate_s_scar(y2, expmt_config['labeling_frequency_g2'])

    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    s = np.concatenate([s1, s2])
    
    for run in range(n_runs):
        x1_train, x1_test, y1_train, y1_test, s1_train, s1_test = train_test_split(x1, y1, s1,
                                                                                   test_size=.4,
                                                                                   random_state=seeds[run],
                                                                                   shuffle=True, stratify=y1)
        x2_train, x2_test, y2_train, y2_test, s2_train, s2_test = train_test_split(x2, y2, s2, 
                                                                                   test_size=.4, 
                                                                                   random_state=seeds[run], 
                                                                                   shuffle=True, stratify=y2)

        x1_val, x1_test, y1_val, y1_test, s1_val, s1_test = train_test_split(x1_test, y1_test, s1_test,  
                                                                             test_size=.5, shuffle=True, stratify=y1_test)
        x2_val, x2_test, y2_val, y2_test, s2_val, s2_test = train_test_split(x2_test, y2_test, s2_test, 
                                                                             test_size=.5, shuffle=True, stratify=y2_test)

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
        
        info = {}
        g1_train_idxs = x_train[:,0] == 1
        g2_train_idxs = x_train[:,1] == 1
        g1_test_idxs = x_test[:,0] == 1
        g2_test_idxs = x_test[:,1] == 1

        classification_attributes = [i + g1_config['n_groups'] for i in range(g1_config['n_attributes'])]

        ### Apply each method to the synthetic data
        if method == 'ours':
            f_model, losses = train_relative_estimator(x_train, s_train, 
                                                       expmt_config, n_epochs=10000)

            pred_rel_prior, pred_g1_prior, pred_g2_prior = eval_relative_prior(x_test, f_model)

            # Evaluate validation set metrics 
            s_preds = torch.squeeze(f_model(torch.Tensor(x_val).cuda()))
            val_loss = get_loss(s_val, s_preds)
            s_preds = torch.squeeze(s_preds).detach().cpu()
            auc = roc_auc_score(s_val, s_preds)
            auprc = average_precision_score(s_val, s_preds)

            info = {'auprc': auprc, 'auc': auc,  'val_loss': val_loss}

        elif method == 'supervised':
            results = supervised_rel_prior(x_train[g1_train_idxs], 
                                           x_train[g2_train_idxs],
                                           y1_train, y2_train, x_test, g1_config)

            pred_rel_prior, pred_g1_prior, pred_g2_prior, _ = results

        elif method == 'negative':
            results = supervised_rel_prior(x_train[g1_train_idxs], 
                                           x_train[g2_train_idxs],
                                           s_train[g1_train_idxs],
                                           s_train[g2_train_idxs], x_test, g1_config)

            pred_rel_prior, pred_g1_prior, pred_g2_prior, _ = results

        elif method == 'sar-em':
            results = sar_em_rel_prior(x_train_norm[g1_train_idxs],
                                       x_train_norm[g2_train_idxs],
                                       s_train[g1_train_idxs],
                                       s_train[g2_train_idxs], x_test_norm, g1_config)

            pred_rel_prior, pred_g1_prior, pred_g2_prior, _ = results

        elif method == 'cdmm':
            results  = cdmm(x_train_norm[g1_train_idxs], x_train_norm[g2_train_idxs],
                        s_train[g1_train_idxs], s_train[g2_train_idxs], g1_config)

            pred_rel_prior, pred_g1_prior, pred_g2_prior, _ = results

        elif method == 'scar-km2':
            results = scar_km2_rel_prior(x1_train, x2_train, s1_train, s2_train)
            pred_rel_prior, pred_g1_prior, pred_g2_prior, _ = results

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
