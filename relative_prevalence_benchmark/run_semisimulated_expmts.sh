 #!/bin/bash

python semisimulated_baseline_comparison.py random supervised 1
#python semisimulated_baseline_comparison.py random scar-km2 1
#python semisimulated_baseline_comparison.py random sar-em 1
#python semisimulated_baseline_comparison.py random negative 1
#python semisimulated_baseline_comparison.py random ours 1

python semisimulated_baseline_comparison.py ipv supervised 1
#python semisimulated_baseline_comparison.py ipv scar-km2 1
#python semisimulated_baseline_comparison.py ipv sar-em 1
#python semisimulated_baseline_comparison.py ipv negative 1
#python semisimulated_baseline_comparison.py ipv ours 1

python semisimulated_baseline_comparison.py corr supervised 1
#python semisimulated_baseline_comparison.py corr scar-km2 1
#python semisimulated_baseline_comparison.py corr sar-em 1
#python semisimulated_baseline_comparison.py corr negative 1
#python semisimulated_baseline_comparison.py corr ours 2

