import pandas as pd
import pdb

ICD_CODE_FIELD = 'icd_code'
mimic_iv_data_path = '/local/divyas/physionet.org/files/mimiciv/1.0/'


# Helper functions for querying IDs 
def get_ids_with_icd_codes(diagnoses, id_type, codes):
    ids = set(diagnoses.loc[diagnoses[ICD_CODE_FIELD].map(lambda x:any([x.startswith(code) 
                                                                        for code in codes])), id_type])
    return ids

def get_ids_with_kws(diagnoses, id_type, kws, descr_field='long_title'):
    ids = set(diagnoses.loc[diagnoses[descr_field].map(lambda x:any([keyword in x.lower() 
                                                                     for keyword in kws])), id_type])
    return ids

def get_ids_of_ethnicity(ids, ethnicity, id_type='hadm_id'):
    # Function that returns idxs of a subset of ids that belong to group
    # specified by attr_name and attr_val
    # attr_name is some option in ['ethnicity']
    admissions = pd.read_csv(mimic_iv_data_path + 'core/admissions.csv.gz')
    ethnicity_ids = admissions[admissions['ethnicity'] == ethnicity][id_type]
    group_ids = sorted(list(set(ids).intersection(set(ethnicity_ids))))
    
    # TODO: seems clunky
    id_to_index = {h_id : idx for idx, h_id in enumerate(ids)}
    group_id_idxs = [id_to_index[g_id] for g_id in group_ids]
    return group_id_idxs
    

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
    
    pdb.set_trace()
    # all diagnoses in general
    all_d = diagnoses.loc[diagnoses[key].map(lambda x:x in sub_d_value_counts.index),key]
    all_d_value_counts = pd.DataFrame(all_d.value_counts())
    all_d_value_counts['proportion_rows'] = all_d_value_counts[key] / len(ids)

    pdb.set_trace()
    
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