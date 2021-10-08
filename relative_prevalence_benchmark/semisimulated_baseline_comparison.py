import sys
import torch 
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, '../')
sys.path.insert(0, '../MIMIC_notebooks')


from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

from baselines import cdmm
from sarpu.pu_learning import *
from sarpu.PUmodels import LogisticRegressionPU
from km.Kernel_MPE_grad_threshold import wrapper as ramaswamy

from gpu_utils import restrict_GPU_pytorch
from simulation_helpers import generate_s_scar
from mimic_helper_fs import get_ids_of_ethnicity
from method import train_relative_estimator, get_loss
from eval_fs import eval_relative_prior, eval_pred_prior
from baselines import cdmm, supervised_rel_prior, sar_em_rel_prior, scar_km2_rel_prior


expmt_mode = sys.argv[1]
methods = [sys.argv[2]]
restrict_GPU_pytorch(sys.argv[3])

if expmt_mode == 'random': 
    split_dir = "../data/semisynthetic/random/1/"
    results_f_name = "results/random_" + methods[0] + "_semisimulated_results"
elif expmt_mode == 'ipv':
    split_dir = "../data/semisynthetic/ipv/1/"
    results_f_name = "results/ipv_" + methods[0] + "_semisimulated_results"
elif expmt_mode == 'corr':
    split_dir = "../data/semisynthetic/corr/endometriosis/0/"
    results_f_name = "results/corr4_" + methods[0] + "_semisimulated_results"
    
result_dicts = []
# read features and labels
x = load_npz(split_dir + 'vals.npz').todense()
y = np.expand_dims(np.loadtxt(split_dir + 'positive_labels'), 1)
hadm_ids = np.loadtxt(split_dir + 'row_names')


# meta experiment config
n_runs = 5
n_attributes = x.shape[1]

c1 = .5 
c2_default = .3
n_groups = 2
penalty = 'l1'
solver = 'liblinear'
fit_intercept = True
scar_assumption = True
seeds = [10, 52, 30, 42, 1000]
classification_model_type = LogisticRegressionPU

c2_opts = np.arange(.1, 1.0, .2)
lamda_opts = [.01, .001, .0001, .00001, .000001]
g1_config = {'n_groups': 2, 'n_attributes': n_attributes}
classification_attributes = np.array(list(range(n_attributes))) + n_groups

### Enumerate experiment configurations

expmt_configs = []
for  c2 in c2_opts:
    for lamda in lamda_opts:
        expmt_config = {'scar_assumption': scar_assumption, 
                        'labeling_frequency_g1': c1, 'labeling_frequency_g2': c2, 
                        'lamda': lamda, 'n_groups': 2,
                        'n_attributes': n_attributes} 
        expmt_configs.append(expmt_config)

g1_hadm_ids_idxs = get_ids_of_ethnicity(hadm_ids, 'BLACK/AFRICAN AMERICAN')
g2_hadm_ids_idxs = get_ids_of_ethnicity(hadm_ids, 'WHITE')

### Subsample to 20% of the data for random and corr experiments

if expmt_mode == 'random' or expmt_mode == 'corr':
    subsample_pct = .2
    g1_hadm_ids_idxs = np.random.choice(g1_hadm_ids_idxs, 
                                        int(subsample_pct*len(g1_hadm_ids_idxs)),
                                        replace=False)
    g2_hadm_ids_idxs = np.random.choice(g2_hadm_ids_idxs, 
                                        int(subsample_pct*len(g2_hadm_ids_idxs)),
                                        replace=False)
x1, y1 = x[g1_hadm_ids_idxs], y[g1_hadm_ids_idxs]
x2, y2 = x[g2_hadm_ids_idxs], y[g2_hadm_ids_idxs] 

### Add group feature to real data 

x1 = np.concatenate([np.ones((len(x1), 1)), np.zeros((len(x1), 1)), x1], axis=1)
x2 = np.concatenate([np.zeros((len(x2), 1)), np.ones((len(x2), 1)), x2], axis=1)

### Run each experiment

for expmt_config in tqdm(expmt_configs):
    s1 = generate_s_scar(y1, expmt_config['labeling_frequency_g1'])
    s2 = generate_s_scar(y2, expmt_config['labeling_frequency_g2'])

    for method in methods:    
        print("METHOD: ", method)
        if method != 'ours' and expmt_config['lamda'] != .01:
            continue
            
        for run in range(n_runs):
            x1_train, x1_test, y1_train, y1_test, s1_train, s1_test = train_test_split(x1, y1, s1,
                                                                                       test_size=.4,
                                                                                       random_state=seeds[run],
                                                                                       shuffle=True, 
                                                                                       stratify=y1)
            x2_train, x2_test, y2_train, y2_test, s2_train, s2_test = train_test_split(x2, y2, s2, 
                                                                                       test_size=.4, 
                                                                                      random_state=seeds[run], 
                                                                                       shuffle=True, 
                                                                                       stratify=y2)

            x1_val, x1_test, y1_val, y1_test, s1_val, s1_test = train_test_split(x1_test, y1_test,
                                                                                 s1_test, test_size=.5,
                                                                                 shuffle=True, 
                                                                                 stratify=y1_test)
            x2_val, x2_test, y2_val, y2_test, s2_val, s2_test = train_test_split(x2_test, y2_test,
                                                                                 s2_test, test_size=.5, 
                                                                                 shuffle=True, 
                                                                                 stratify=y2_test)
            
            x_train = np.concatenate([x1_train, x2_train])
            y_train = np.concatenate([y1_train, y2_train])
            s_train = np.concatenate([s1_train, s2_train])
            
            x_val = np.concatenate([x1_val, x2_val])
            s_val = np.concatenate([s1_val, s2_val])
            x_test = np.concatenate([x1_test, x2_test])

            y1_train = y1_train.squeeze(axis=1)
            y2_train = y2_train.squeeze(axis=1)
            s1_train = s1_train.squeeze(axis=1)
            s2_train = s2_train.squeeze(axis=1)

            scale_factor = np.max(np.abs(x_train))
            x_train_norm = x_train/scale_factor
            x_val_norm = x_val/scale_factor
            x_test_norm = x_test/scale_factor
           
            g1_train_idxs = np.squeeze(np.array(x_train[:,0] == 1))
            g2_train_idxs = np.squeeze(np.array(x_train[:,1] == 1))
            g1_test_idxs = np.squeeze(np.array(x_test[:,0] == 1))
            g2_test_idxs = np.squeeze(np.array(x_test[:,1] == 1))
            info = {}           
            if method == 'ours':
                x_train = np.concatenate([x1_train, x2_train])
                s_train = np.concatenate([s1_train, s2_train])
                f_model, losses = train_relative_estimator(x_train, s_train, expmt_config, 
                                                           n_epochs=20000, save=True)
                
                pred_rel_prior, pred_g1_prior, pred_g2_prior = eval_relative_prior(x_test, f_model)

                s_preds = f_model(torch.Tensor(x_val).cuda())
                val_loss = get_loss(s_val, s_preds)
                s_preds = torch.squeeze(s_preds).detach().cpu()
                auc = roc_auc_score(s_val, s_preds)
                auprc = average_precision_score(s_val, s_preds)

                info = {'auprc': auprc,
                        'auc': auc, 
                        'val_loss': val_loss}
            else:
                classification_model = classification_model_type(fit_intercept=fit_intercept,
                                                                 penalty=penalty, solver=solver)
                if method == 'supervised':
                    results = supervised_rel_prior(x_train[g1_train_idxs], 
                                                   x_train[g2_train_idxs],
                                                   y1_train, y2_train, x_test, g1_config,
                                                   classification_model=classification_model)

                    pred_rel_prior, pred_g1_prior, pred_g2_prior, _ = results
   
                elif method == 'negative':
                    results = supervised_rel_prior(x_train[g1_train_idxs], 
                                                   x_train[g2_train_idxs],
                                                   s_train[g1_train_idxs],
                                                   s_train[g2_train_idxs], x_test, g1_config,
                                                   classification_model=classification_model)

                    pred_rel_prior, pred_g1_prior, pred_g2_prior, _ = results
                
                elif method == 'sar-em':
                    results = sar_em_rel_prior(x_train_norm[g1_train_idxs],
                                               x_train_norm[g2_train_idxs],
                                               s_train[g1_train_idxs],
                                               s_train[g2_train_idxs], x_test_norm, g1_config,
                                               classification_model=classification_model)

                    pred_rel_prior, pred_g1_prior, pred_g2_prior, _ = results
                
                
                elif method == 'scar-km2':
                    results = scar_km2_rel_prior(x1_train, x2_train, s1_train, s2_train)
                    pred_rel_prior, pred_g1_prior, pred_g2_prior, _ = results
                            
                elif method == 'cdmm':
                    info = {}
                    pred_rel_prior, pred_g1_prior, pred_g2_prior = cdmm(x_train[g1_train_idxs],
                                                                        x_train[g2_train_idxs],
                                                                        s_train[g1_train_idxs],
                                                                        s_train[g2_train_idxs],
                                                                        g1_config)
            true_g1_prior = y1_test.mean()
            true_g2_prior = y2_test.mean()
            true_rel_prior = true_g1_prior / true_g2_prior
                            
            result_dict = {'pred_rel_prior': pred_rel_prior, 'true_rel_prior': true_rel_prior, 
                           'rel_prior_err': true_rel_prior - pred_rel_prior,
                           'pred_g1_prior': pred_g1_prior, 'pred_g2_prior': pred_g2_prior,
                           'true_g1_prior': true_g1_prior, 'true_g2_prior': true_g2_prior,
                           'method': method, 'run': run}
            result_dict.update(expmt_config)
            result_dict.update(info)
            result_dicts.append(result_dict)

        results_df = pd.DataFrame(result_dicts)
        results_df.to_csv(results_f_name)
