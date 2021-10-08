# invisible_conditions
Methods for quantifying prevalence of underreported medical conditions like IPV. 

### Download data and generate semi-synthetic datasets.

> 1. Download data from [MIMIC-IV](https://physionet.org/content/mimiciv/0.4/) and [MIMIC-IV ED](https://physionet.org/content/mimic-iv-ed/1.0/). You will need to complete training in order to access both.
> 2. Set paths for MIMIC-IV and MIMIC-IV ED in the file ./MIMIC_notebooks/mimic_paths.py
> 3. Run each of the following notebooks ("Generate Random Semi-Simulated Dat", "Generate Endometriosis Correlation Data", "Generate IPV Semi-Simulated Data") to generate the datasets required for the semi-synthetic experiments.

### Install required libraries.

> 1. Clone the [SAR-PU library](https://github.com/ML-KULeuven/SAR-PU) into root folder, and follow the repo's instructions to install the required libraries.
> 2. Clone the [PU_class_prior library](https://github.com/teisseyrep/PU_class_prior.git) into the root folder. This is for the baseline CDMM. 
> 3. Create a conda environment and install the packages in requirements.txt by running 
> ``` conda create -n <environment-name> --file requirements.txt ```

### Reproduce experiments.

> Run the following: 
> ```
> cd relative_prevalence_benchmark
> ./run_simulated_experiments.sh
> ./run_semisimulated_experiments.sh
> ./run_robustness_experiments.sh
> ```
> You can generate each of the figures and tables in the paper using the notebooks in the ```./relative_prevalence_benchmark/``` folder. Let us know if you run into any issues by emailing divyas@mit.edu or filing a Github issue! 
