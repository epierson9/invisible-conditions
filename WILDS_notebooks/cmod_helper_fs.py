import numpy as np
import pdb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = '/local/divyas/invisible_conditions'
MODE_GROUP_MAP_WITH_OTHER = {'gender': ('male', 'female', 'transgender', 'other'),
                  'religion': ('christian', 'muslim', 'other_religions', 'other'),
                  'sexual_orientation': ('heterosexual', 'homosexual_gay_or_lesbian', 'other'), 
                  'identity': ('black', 'white', 'asian', 'latino', 'other')}

MODE_GROUP_MAP = {'gender': ('male', 'female', 'transgender'),
                  'religion': ('christian', 'muslim', 'jewish', 'other_religions'),
                  'sexual_orientation': ('heterosexual', 'homosexual_gay_or_lesbian'), 
                  'identity': ('black', 'white', 'asian', 'latino')}

group_metadata_columns = ['male', 'female', 'LGBTQ', 'homosexual_gay_or_lesbian',
                          'heterosexual', 'transgender', 'christian', 'jewish', 'muslim',
                          'other_religions', 'black', 'white', 'asian', 'latino']

prettify_group_name = {'male': 'Male',
                       'female': 'Female',
                       'transgender': 'Transgender',
                       'christian': 'Christian',
                       'muslim': 'Muslim',
                       'jewish': 'Jewish',
                       'other_religions': 'Other Religions',
                       'heterosexual': 'Heterosexual',
                       'homosexual_gay_or_lesbian': 'Homosexual',
                       'black': 'Black',
                       'white': 'White',
                       'asian': 'Asian',
                       'latino': 'Latino'}

prettify_category_name = {'gender': 'Gender',
                       'religion': 'Religion',
                       'sexual_orientation': 'Sexual Orientation',
                        'identity': 'Identity'}

def load_cmod_metadata():
    # TODO: make this not hard coded
    metadata = np.load(DATA_PATH + "/data/real_true_labels/content_mod/0.5/metadata.npy")
    metadata_columns = np.array(np.load(DATA_PATH + "/data/real_true_labels/content_mod/0.5/metadata_col_names.npy"))
    return metadata, metadata_columns

metadata, metadata_columns = load_cmod_metadata()


def get_group_idx_in_metadata(group):
    col_idx = np.where(metadata_columns == group)[0][0]
    return col_idx

def get_idxs_of_group_cmod(group):
    col_idx = get_group_idx_in_metadata(group)
    return np.where(metadata[:,col_idx] == 1)[0]

def get_idxs_not_of_group_cmod(group):
    col_idx = get_group_idx_in_metadata(group)
    return np.where(metadata[:,col_idx] == 0)[0]

def get_idxs_of_no_group_cmod():
    all_groups = group_metadata_columns 
    col_idxs = [get_group_idx_in_metadata(group) for group in all_groups]
    return np.where(metadata[:,col_idxs].sum(axis=1) == 0)[0]

def get_idxs_not_of_groups_cmod(groups):
    col_idxs = [get_group_idx_in_metadata(group) for group in groups]
    groups_indicator = np.sum(metadata[:,col_idxs], axis=1)
    return np.where(groups_indicator == 0)[0]

def generate_s_scar(y, labeling_freq):
    y_pos_idxs = np.where(y == 1)[0]
    labeled_idxs = np.random.choice(y_pos_idxs, int(len(y_pos_idxs)*labeling_freq), replace=False)
    s = np.zeros(y.shape)
    s[labeled_idxs] = 1
    return s

def get_group_idxs_list(groups):
    group_idxs_list = []
    for group in groups:
        if group != 'other':
            group_idxs = get_idxs_of_group_cmod(group)
        else:  
#             group_idxs = get_idxs_not_of_groups_cmod([g for g in groups if g != 'other'])
            group_idxs = get_idxs_of_no_group_cmod()
        print(group, len(group_idxs))
        group_idxs_list.append(group_idxs)
    
    return group_idxs_list

def preprocess_cmod_data(x, y, s, text, toxicity, groups, expmt_config, random_seed=0, 
                         include_category_complement=False):
    # If s=[], simulated based on the labeling frequencies in expmt_config
    # If y=[], set as a dummy vector of -1s 

    n_groups = len(groups)
    label_frequencies = expmt_config['labeling_frequencies']
    group_to_idx = {group: i for i,group in enumerate(groups)}
    group_idxs_list = get_group_idxs_list(groups)
    
        
    group_data_tuples = []
    # Add group features to x 
    x = np.concatenate([np.zeros((x.shape[0], n_groups)), x], axis =1)
    for i, group_idxs in enumerate(group_idxs_list):
        x[group_idxs,i] = 1
        
    # Drop examples that have multiple or none of the group labels    
    x = pd.DataFrame(x)
    x['comment_text'] = text
    print("# of Comments: ", len(x))
    x['toxicity'] = toxicity
    x['y'] = y if len(y) else -1*np.ones(len(x))
    x['s'] = s if len(s) else -1*np.ones(len(x))
    x['group_membership'] = x[list(range(n_groups))].sum(1)
    x = x[x['group_membership'] == 1]
    print("# of Comments in 1 Group: ", len(x), '\n')
    # Add s to dataframe, if it must be simulated
    if not len(s):
        s = np.zeros(len(x))
        for i in range(n_groups):
            group_c = label_frequencies[i]
            group_idxs = np.where(x[i] == 1)[0]
            group_y = x[x[i] == 1]['y']
            group_s = generate_s_scar(group_y, group_c)
            s[group_idxs] = group_s
        x['s'] = s

    # Stratify by groups & by toxicity label
    x['stratify_var'] = x[list(range(n_groups)) + ['s']].astype(str).agg('-'.join, axis=1)
    train, test = train_test_split(x, test_size=.4, stratify=x['stratify_var'])
    val, test = train_test_split(test, test_size=.5, stratify=test['stratify_var'])

    # Output is processed x, s, y, and text representations for each example 
    x_train, x_val, x_test = [np.array(d)[:,:d.shape[1]-6].astype('float64') for d in (train, val, test)]
    s_train, s_val, s_test = [np.array(d['s']) for d in (train, val, test)]
    y_train, y_val, y_test = [np.array(d['y']) for d in (train, val, test)]
    tox_train, tox_val, tox_test = [np.array(d['toxicity']) for d in (train, val, test)]
    t_train, t_val, t_test = [list(d['comment_text']) for d in (train, val, test)]
    return (x_train, y_train, s_train, t_train, tox_train), (x_val, y_val, s_val, t_val, tox_val), (x_test, y_test, s_test, t_test, tox_test)

def normalize_x(x_train, x_val, x_test, normalize_groups=False, n_groups=2):
    scaler =  StandardScaler()
    if normalize_groups:
        scaler.fit(x_train)
        x_train_norm, x_val_norm, x_test_norm = [scaler.transform(d) for d in (x_train, x_val, x_test)]
    else:
        scaler.fit(x_train[:,n_groups:])
        x_train_norm, x_val_norm, x_test_norm = [scaler.transform(d[:,n_groups:]) for d in (x_train, x_val, x_test)]
        norm_ds = []
        for norm_d, d in [(x_train_norm, x_train), (x_val_norm, x_val), (x_test_norm, x_test)]:
            norm_d_with_group = np.concatenate([d[:,:n_groups], norm_d], axis=1)
            norm_ds.append(norm_d_with_group)
        x_train_norm, x_val_norm, x_test_norm = norm_ds
    return (x_train_norm, x_val_norm, x_test_norm)

def get_group_weights(x, n_groups):
    group_weights = np.sum(x[:,:n_groups], axis=0)
    group_weights = np.max(group_weights) / group_weights
    group_weights = group_weights/np.sum(group_weights)
    group_weights = np.array(group_weights).flatten()
    return group_weights
