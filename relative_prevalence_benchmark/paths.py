import os

MODEL_DIR = '/local/divyas/invisible_conditions/relative_prevalence_benchmark/model_ckpts/'
RESULTS_DIR = '/local/divyas/invisible_conditions/relative_prevalence_benchmark/results/'
FIG_DIR = '/local/divyas/invisible_conditions/relative_prevalence_benchmark/figs/'

for dirname in [MODEL_DIR, RESULTS_DIR, FIG_DIR]:
    os.makedirs(dirname, exist_ok=True)


