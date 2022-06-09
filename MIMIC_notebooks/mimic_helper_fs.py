import pandas as pd
import numpy as np
import pdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mimic_paths import mimic_iv_path

ICD_CODE_FIELD = 'icd_code'
mimic_iv_data_path = mimic_iv_path
MODE_GROUP_MAP = {'ethnicity': ('BLACK/AFRICAN AMERICAN', 'WHITE', 'ASIAN', 'HISPANIC/LATINO'),
                  'insurance': ('Medicaid', 'Medicare'),
                  'marital_status': ('DIVORCED', 'MARRIED', 'SINGLE')}

prettify_group_name = {'BLACK/AFRICAN AMERICAN': 'Black/African American',
                        'WHITE': 'White',
                        'ASIAN': 'Asian',
                        'HISPANIC/LATINO': 'Hispanic/Latino',
                        'F': 'Female',
                        'M': 'Male',
                        'Medicaid': 'Medicaid',
                        'Medicare': 'Medicare',
                        'DIVORCED': 'Divorced',
                        'MARRIED': 'Married',
                        'SINGLE': 'Single'}

prettify_category_name = {'ethnicity': 'Ethnicity',
                          'insurance': 'Insurance',
                          'marital_status': 'Marital Status'}
                        
# Helper functions for querying IDs 
def get_ids_with_icd_codes(diagnoses, id_type, codes):
    ids = set(diagnoses.loc[diagnoses[ICD_CODE_FIELD].map(lambda x:any([x.startswith(code) 
                                                                        for code in codes])), id_type])
    return ids

def get_ids_with_kws(diagnoses, id_type, kws, descr_field='long_title'):
    ids = set(diagnoses.loc[diagnoses[descr_field].map(lambda x:any([keyword in x.lower() 
                                                                     for keyword in kws])), id_type])
    return ids

def get_idxs_of_group(ids, group_name, category, id_type='hadm_id'):
    # Function that returns idxs of a subset of ids that belong to group
    # specified by attr_name and attr_val
    # attr_name is some option in ['ethnicity']
    admissions = pd.read_csv(mimic_iv_data_path + 'core/admissions.csv.gz')
    group_ids = admissions[admissions[category] == group_name][id_type]
    group_ids = sorted(list(set(ids).intersection(set(group_ids))))
    
    id_to_index = {h_id : idx for idx, h_id in enumerate(ids)}
    group_id_idxs = [id_to_index[g_id] for g_id in group_ids]
    return group_id_idxs

def get_idxs_not_of_group(ids, group_name, category, id_type='hadm_id'):
    group_idxs = get_idxs_of_group(ids, group_name, category, id_type)
    all_idxs = list(range(len(ids)))
    not_group_idxs = sorted(list(set(all_idxs).difference(set(group_idxs))))
    return not_group_idxs
                    
#     not_group_ids = sorted(list(set(ids).difference(set(group_ids))))
#     pdb.set_trace()                       
#     id_to_index = {h_id : idx for idx, h_id in enumerate(ids)}
#     not_group_id_idxs = [id_to_index[g_id] for g_id in not_group_ids]
#     return not_group_id_idxs
                           
def get_ids_of_ethnicity(ids, ethnicity, id_type='hadm_id'):
    return get_ids_of_group(ids, ethnicity, 'ethnicity', id_type='hadm_id')

def preprocess_mimic_data(x, y, s, hadm_ids, subject_ids, groups, expmt_config, random_seed=0):
    if len(hadm_ids.shape) == 2:
        hadm_ids = np.squeeze(hadm_ids, 1)
    if len(hadm_ids.shape) == 2: 
        subject_ids = np.squeeze(subject_ids, 1)
    if not len(y):
        y = s
   
    
    # Create group indicator columns
    n_groups = len(groups)
    group_indicators_list = []
    for group in groups:
        group_indicators = np.zeros((hadm_ids.shape[0], 1))
        group_idxs = get_idxs_of_group(hadm_ids, group, expmt_config['category'])
        group_indicators[group_idxs] = 1
        group_indicators_list.append(group_indicators)
    
    # Add group features to x
    x = np.concatenate([*group_indicators_list, x], axis=1)
    
    group_columns = list(range(n_groups))
    columns = group_columns + ['hadm_id', 'subject_id', 's']
    hadm_id_columns = np.stack([[x.flatten() for x in group_indicators_list] + [hadm_ids, subject_ids, s]])[0]
    hadm_id_df = pd.DataFrame(hadm_id_columns.T,  columns=columns)
    
    # Split patients into train, val, and test sets
    subject_id_df = hadm_id_df[['subject_id', 's'] + group_columns].copy()
    subject_id_df = subject_id_df.drop_duplicates(subset='subject_id', keep='first')
    subject_id_df['dummy'] = 1
    if expmt_config['stratify_mode'] == 'group_and_s':
        subject_id_df['stratify_var'] = subject_id_df[group_columns + ['s']].astype(str).agg('-'.join, axis=1)
    elif expmt_config['stratify_mode'] == 's':
        subject_id_df['stratify_var'] = subject_id_df['s']
    elif expmt_config['stratify_mode'] == 'group':
        subject_id_df['stratify_var'] = subject_id_df[group_columns + ['s']].astype(str).agg('-'.join, axis=1)
    elif expmt_config['stratify_mode'] == 'none':
        subject_id_df['stratify_var'] = subject_id_df['dummy']

    train_sids, test_sids = train_test_split(subject_id_df.subject_id,
                                                              test_size=.4,
                                                              shuffle=True, random_state=random_seed,
                                                              stratify=subject_id_df.stratify_var)
    test_sid_df = subject_id_df[subject_id_df['subject_id'].isin(test_sids)]
    val_sids, test_sids = train_test_split(test_sid_df.subject_id, test_size=.5, 
                                           shuffle=True, random_state=random_seed,
                                           stratify=test_sid_df.stratify_var)
    
    # Map patient IDs to hospital admission IDs
    train_hadm_ids = hadm_id_df[hadm_id_df['subject_id'].isin(train_sids)]['hadm_id']
    val_hadm_ids = hadm_id_df[hadm_id_df['subject_id'].isin(val_sids)]['hadm_id']
    test_hadm_ids = hadm_id_df[hadm_id_df['subject_id'].isin(test_sids)]['hadm_id']

    # Map hospital admission IDs to indices in the original input, x
    hadm_id_to_idx = {h_id : idx for idx, h_id in enumerate(hadm_ids)}
    train_idxs = [hadm_id_to_idx[hadm_id] for hadm_id in list(train_hadm_ids)]
    val_idxs = [hadm_id_to_idx[hadm_id] for hadm_id in list(val_hadm_ids)]
    test_idxs = [hadm_id_to_idx[hadm_id] for hadm_id in list(test_hadm_ids)]

    # Define objects for train, val, and test set 
    train = x[train_idxs], y[train_idxs], s[train_idxs]
    val = x[val_idxs], y[val_idxs], s[val_idxs]
    test = x[test_idxs],  y[test_idxs], s[test_idxs]
    return train, val, test

def normalize_x(x_train, x_val, x_test, normalize_groups=False, n_groups=2):
    scaler = StandardScaler()
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

# Helper functions for querying symptoms via ICD codes or keywords
def get_icd_code_long_title(names, code):
    code_names = names[names['icd_code'].str.startswith(code)]['long_title']
    return list(code_names)[0]

def get_icd_codes_with_prefix(names, code):
    code_names = names[names['icd_code'].str.startswith(code)]['icd_code']
    return list(code_names)

def get_coocurring_symptoms_codes(diagnoses, id_type, codes, key='long_title', print_output=True):
    assert type(codes) is list
    assert id_type in ['hadm_id', 'subject_id']
    # ids with ICD9 codes that contain any keyword in query
    ids = get_ids_with_icd_codes(diagnoses, id_type, codes)
    # all diagnoses associated with ids 
    sub_d = diagnoses.loc[diagnoses[id_type].map(lambda x:x in ids), key]    
    
    # filter for frequent diagnoses
    sub_d_value_counts = pd.DataFrame(sub_d.value_counts().head(n=50))
    sub_d_value_counts['proportion_rows'] = sub_d_value_counts[key] / len(ids)
    sub_d_value_counts.columns = ['# rows', '# rows/# IDs']
    
    # all diagnoses in general
    all_d = diagnoses.loc[diagnoses[key].map(lambda x:x in sub_d_value_counts.index),key]
    all_d_value_counts = pd.DataFrame(all_d.value_counts())
    all_d_value_counts['proportion_rows'] = all_d_value_counts[key] / len(ids)
    
    if print_output:
        print("# Codes: %s, %s. Total IDs: %i; total diagnoses: %i" % (len(codes), id_type, len(ids), len(sub_d)))
        print(value_counts)
    return ids, sub_d, value_counts

def get_coocurring_symptoms_cpt_codes(diagnoses, cpt_events, id_type, codes, print_output=True):
    assert type(codes) is list
    assert id_type in ['hadm_id', 'subject_id']
    # ids with ICD9 codes that contain any keyword in query
    ids = get_ids_with_icd_codes(diagnoses, id_type, codes)
    # all diagnoses associated with ids 
    sub_d = cpt_events.loc[cpt_events[id_type].map(lambda x:x in ids), 'cpt_number']
    
    # filter for frequent diagnoses
    value_counts = pd.DataFrame(sub_d.value_counts().head(n=20))
    value_counts['proportion_rows'] = value_counts['cpt_number'] / len(ids)
    value_counts.columns = ['# rows', '# rows/# IDs']
    if print_output:
        print("# Codes: %s, %s. Total IDs: %i; total diagnoses: %i" % (len(codes), id_type, len(ids), len(sub_d)))
        print(value_counts)
    
    # TODO: fix how this function returns sth different from others
    return ids, sub_d, value_counts


def get_coocurring_symptoms_kws(diagnoses, id_type, query, print_output=True):
    assert type(query) is list
    assert id_type in ['hadm_id', 'subject_id']
    # ids with ICD9 codes that contain any keyword in query
    ids = get_ids_with_kws(diagnoses, id_type, query)
    # all diagnoses associated with ids 
    sub_d = diagnoses.loc[diagnoses[id_type].map(lambda x:x in ids), 'long_title']
    
    # filter for frequent diagnoses
    value_counts = pd.DataFrame(sub_d.value_counts().head(n=20))
    
    value_counts['proportion_rows'] = value_counts['long_title'] / len(ids)
    value_counts.columns = ['# rows', '# rows/# IDs']
    if print_output:
        print("Query: %s, %s. Total IDs: %i; total diagnoses: %i" % (query, id_type, len(ids), len(sub_d)))
        print(value_counts)
    return ids, sub_d
