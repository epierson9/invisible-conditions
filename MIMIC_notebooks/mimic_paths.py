mimic_iv_path = '~/data/physionet.org/files/mimiciv/1.0/'
ed_path = '~/data/physionet.org/files/mimic-iv-ed/1.0/ed/'
hosp_path = mimic_iv_path + 'hosp/'
icu_path = mimic_iv_path + 'icu/'

admissions_path = mimic_iv_path + 'core/admissions.csv.gz'
patients_path = mimic_iv_path + 'core/patients.csv.gz'

hosp_diagnoses_path = hosp_path + 'diagnoses_icd.csv.gz'
ed_diagnoses_path = ed_path + 'diagnosis.csv.gz'
input_events_path = icu_path + '/inputevents.csv.gz'
items_path = icu_path + '/d_items.csv.gz'
chart_events_path = icu_path + '/chartevents.csv.gz'

english_names_path = hosp_path + 'd_icd_diagnoses.csv.gz'
