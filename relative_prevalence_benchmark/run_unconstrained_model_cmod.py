import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '../')
sys.path.insert(0, '../WILDS_notebooks/')
import torch
from gpu_utils import restrict_GPU_pytorch
from method import train_relative_estimator, load_model
from paths import RESULTS_DIR
from cmod_helper_fs import preprocess_cmod_data,  MODE_GROUP_MAP_WITH_OTHER, normalize_x
from cmod_helper_fs import get_group_weights
import pdb
restrict_GPU_pytorch(sys.argv[1])

seeds = [10, 52, 30, 42, 1000]

include_complement = True
results_f_name = "results/"
categories = MODE_GROUP_MAP_WITH_OTHER.keys()
    
# read features and labels
split_dir = "../data/real_true_labels/content_mod/0.5/"
x = np.load(split_dir + 'vals.npz.npy')
s = np.load(split_dir + 'observed_labels.npy')
text = np.load(split_dir + 'text.npy')
toxicity = np.load(split_dir + 'toxicity.npy')

# meta experiment config
n_runs = 5
n_attributes = x.shape[1]

penalty = 'l1'
solver = 'liblinear'
fit_intercept = True
scar_assumption = True
stratify_mode = 'none'
expmt_configs = []
for seed in seeds:
    for category in categories:
        groups = MODE_GROUP_MAP_WITH_OTHER[category]
        n_groups = len(groups)

        expmt_config = {'scar_assumption': scar_assumption, 
                    'lamda': .0001, 'n_groups': n_groups,
                    'n_attributes': n_attributes,'groups': groups,
                    'labeling_frequencies': (None, None),
                    'category': category, 'n_epochs': 10000,
                    'optimizer': 'Adam', 
                    'seed': seed, 
                    'estimator_type': 'unconstrained', 'n_batches': 5,
                    'mode': 'unnorm', 'stratify_mode': stratify_mode}
        expmt_configs.append(expmt_config)


results = []
results_path = RESULTS_DIR + '/unconstrained_model_cmod'
for expmt_config in expmt_configs:
    train, val, test = preprocess_cmod_data(x, y=[], s=s, text=text, toxicity=toxicity,
                                            groups=expmt_config['groups'], 
                                            expmt_config=expmt_config, random_seed=expmt_config['seed'])
    # Output is processed x, s, and y and group_idxs 
    x_train, y_train, s_train, t_train, tox_train = train
    x_val, y_val, s_val, t_val, tox_val = val
    x_test, y_test, s_test, t_test, tox_test = test

    x_train_norm, x_val_norm, x_test_norm = normalize_x(x_train, x_val, x_test, 
                                                        n_groups=expmt_config['n_groups'])

    expmt_config['group_weights'] = get_group_weights(x_train, expmt_config['n_groups'])

    train_data =  x_train
    val_data = x_val
    test_data = x_test
    if expmt_config['mode'] == 'norm':
        train_data = x_train_norm
        val_data = x_val_norm
        test_data = x_test_norm

    f_model_unconstrained, losses, info= train_relative_estimator(train_data, s_train, val_data, s_val, 
                                                        expmt_config, save_model=True)
    
    model_preds = f_model_unconstrained(torch.Tensor(test_data).cuda()).detach().cpu().numpy().flatten()
    result = {'auc': info['auc'], 'auprc': info['auprc'], 'category': expmt_config['category'],
              'seed': expmt_config['seed']}
    print("RESULT: ", result)
    results.append(result)
    pd.DataFrame(results).to_csv(results_path)
