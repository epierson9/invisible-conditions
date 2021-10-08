 #!/bin/bash

python simulated_baseline_comparison.py separability supervised 1
#python simulated_baseline_comparison.py separability sar-em 1
#python simulated_baseline_comparison.py separability negative 1
#python simulated_baseline_comparison.py separability ours 1
#python simulated_baseline_comparison.py separability scar-km2 1


python simulated_baseline_comparison.py group_gap supervised 1
#python simulated_baseline_comparison.py group_gap negative 1
#python simulated_baseline_comparison.py group_gap sar-em 1
#python simulated_baseline_comparison.py group_gap ours 1
#python simulated_baseline_comparison.py group_gap scar-km2 1

python simulated_baseline_comparison.py label_freq supervised 1
#python simulated_baseline_comparison.py label_freq negative 1
#python simulated_baseline_comparison.py label_freq sar-em 1
#python simulated_baseline_comparison.py label_freq ours 1
#python simulated_baseline_comparison.py label_freq scar-km2 1
