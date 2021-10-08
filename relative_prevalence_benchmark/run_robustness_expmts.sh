#!/bin/bash

python simulated_robustness_comparison.py supervised 1
python simulated_robustness_comparison.py scar-km2 1
python simulated_robustness_comparison.py sar-em 1
python simulated_robustness_comparison.py negative 1
python simulated_robustness_comparison.py ours 1
python simulated_robustness_comparison.py cdmm 1

