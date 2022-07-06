# invisible_conditions
Methods for quantifying prevalence of underreported medical conditions like IPV. 

### Download and preprocess data.
1. Intimate Partner Violence data. 
    1. Download data from [MIMIC-IV](https://physionet.org/content/mimiciv/0.4/) and [MIMIC-IV ED](https://physionet.org/content/mimic-iv-ed/1.0/). You will need to complete training in order to access both.
     2. Set paths for MIMIC-IV and MIMIC-IV ED in the file ```./MIMIC_notebooks/mimic_paths.py```.
     3. Generate  each of the semi-synthetic dataset by running each of the following notebooks: "Generate Random Semi-Simulated Data.ipynb", "Generate Endometriosis Correlation Data.ipynb", "Generate IPV Semi-Simulated Data.ipynb", all under ```./MIMIC_notebooks```.
     4. Generate the real dataset by running the following notebook: "./MIMIC_notebooks/Generate Real IPV Data.ipynb".
2. Content Moderation data.
     1. Download the pre-trained model, trained via ERM, from [this link](https://worksheets.codalab.org/rest/bundles/0xb820ddc4bdc44c0d9e298c0eb51335a3/contents/blob/best_model.pth). Store "best_model.pth" in the ```./WILDS_notebooks``` folder.
    2. Generate the CivilComments dataset by running the following notebook: "./WILDS_notebooks/Preprocess CivilComments Dataset.ipynb".
   3. Set global paths to the results, models, and figure directories in ```relative_prevalence_benchmark/paths.py```.

### Install required libraries.

 1. Clone the [SAR-PU library](https://github.com/ML-KULeuven/SAR-PU) into root folder, and follow the repo's instructions to install the required libraries.
 2. Create a conda environment and install the packages in requirements.txt by running 
 ``` conda create -n <environment-name> --file requirements.txt ```

### Reproduce experiments and figures.

 To reproduce experiments in the main text, run the following: 
 ```
 cd relative_prevalence_benchmark
 ./run_synthetic_expmts.sh
 ./run_semisynthetic_expmts.sh
 ./run_mimic_expmts.sh
 ./run_cmod_expmts.sh
 ```
 To reproduce experiments in the supplement, run the following:
 ```
 cd relative_prevalence_benchmark
 ./run_unconstrained_models.sh
 ./run_synthetic_cmod_expmts.sh
 ```
 You can generate each of the figures and tables in the paper using the notebooks in ```./relative_prevalence_benchmark/```; each notebook title includes the figure numbers it reproduces. A short guide to the files in ```./relative_prevalence_benchmark``` is that  ```benchmark*.py``` files implement a comparison of PURPLE to other baselines for a specific dataset, where different files differ by the datasets they use. All files of the form ```run*.sh``` are experiment scripts, meant to fire off all comparisons for a particular  experiment. 
  
Email divyas@mit.edu if you run into any issues!
