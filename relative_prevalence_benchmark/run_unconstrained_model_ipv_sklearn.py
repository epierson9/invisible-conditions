import sys
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

sys.path.insert(0, '../')
sys.path.insert(0, '../MIMIC_notebooks/')
from mimic_helper_fs import get_icd_code_long_title, preprocess_mimic_data, MODE_GROUP_MAP, normalize_x, get_group_weights
from paths import RESULTS_DIR

### Load data
data_dir = "../data/real_true_labels/hospital/ipv/0/"
hadm_ids = np.loadtxt(data_dir + "row_names")
subject_ids = np.loadtxt(data_dir + 'subject_ids')
feature_icd_codes = list(open(data_dir + "feat_names").read().split('\n'))
x = load_npz(data_dir + 'vals.npz').todense()
s = np.loadtxt(data_dir + "observed_labels")

mimic_iv_data_path = '~/data/physionet.org/files/mimiciv/1.0/'
hosp_data_path = mimic_iv_data_path + 'hosp/'
english_names = pd.read_csv(hosp_data_path + 'd_icd_diagnoses.csv.gz')
stratify_mode = 'none'
seeds = [10, 52, 30, 42, 1000] 

### Enumerate experiment configurations
categories = ['insurance', 'ethnicity', 'marital_status']
n_runs = 5
n_attributes = x.shape[1]
n_epochs = 5001

seeds = [10, 52, 30, 42, 1000]
lamda_opts = [0, 1e-6, 1e-5, 1e-4]

expmt_configs = []
for run in range(n_runs):
    for category in categories:
        form lamda in lamda_opts:
            groups = MODE_GROUP_MAP[category]
            n_groups = len(groups)
            classification_attributes = np.array(list(range(n_attributes))) + n_groups

            expmt_config = {'scar_assumption': True, 
                            'lamda': lamda, 'n_groups': n_groups,
                            'n_attributes': n_attributes, 'groups': groups, 
                            'labeling_frequencies': (None, None),
                            'category': category, 'n_epochs': n_epochs, 
                            'optimizer': 'Adam',
                            'seed': seeds[run], 
                            'estimator_type': 'sklearn_unconstrained', 'n_batches': 30, 
                            'stratify_mode': stratify_mode,
                            'mode': 'unnorm'} 
            expmt_configs.append(expmt_config)

results_path = RESULTS_DIR + 'unconstrained_ipv_results_sklearn'
results = []

def add_group_interactions(data, n_groups):
    transformed_data = []
    transformed_data.append(data[:,n_groups:])
    for i in range(n_groups):
        group_interaction = np.multiply(data[:,i],data[:,n_groups:])
        transformed_data.append(group_interaction)
    transformed_data = np.concatenate(transformed_data, axis=1)
    return transformed_data

# Create data with group interactions

for expmt_config in tqdm(expmt_configs):
    seed = expmt_config['seed']
    category = expmt_config['category']
    lamda = expmt_config['lamda']
    groups = MODE_GROUP_MAP[category]        
    expmt_config['group_weights'] = get_group_weights(x, expmt_config['n_groups'])
    print("Unconstrained model for: ", category, "  seed: ", seed)
    ### Split data
    print("Loading data...")
    train, val, test = preprocess_mimic_data(x, [], s, hadm_ids, subject_ids,
                                                 groups, expmt_config, seed)

    x_train, y_train, s_train = train
    x_val, y_val, s_val = val
    x_test, y_test, s_test = test

    
    x_train_norm, x_val_norm, x_test_norm = normalize_x(x_train, x_val, x_test)
    
    ### Choose normalized or unnormalized  data
    train_data = x_train_norm
    val_data = x_val_norm
    test_data = x_test_norm
    if expmt_config['mode'] != 'norm':
        train_data =  x_train
        val_data = x_val
        test_data = x_test

    
    transformed_train_data = add_group_interactions(train_data, expmt_config['n_groups'])
    transformed_val_data = add_group_interactions(val_data, expmt_config['n_groups'])
    transformed_test_data = add_group_interactions(test_data, expmt_config['n_groups'])
    
    lr = LogisticRegression(C=1/lamda)
    lr.fit(transformed_train_data, s_train)
    preds = lr.predict_proba(transformed_test_data)[:,1]
    auc = roc_auc_score(s_test, preds)
    auprc = average_precision_score(s_test, preds)

    

    
    result = {'auc': auc, 'auprc': auprc, 'category': category, 'seed': seed, 
              'lamda': lamda, 'split': 'val'}
    result.update(expmt_config)
    print("RESULT: ", result)
    results.append(result)
    pd.DataFrame(results).to_csv(results_path)