 #!/bin/bash

GPU=1

python benchmark_synthetic.py separability supervised $GPU
python benchmark_synthetic.py separability sar-em $GPU
python benchmark_synthetic.py separability negative $GPU
python benchmark_synthetic.py separability ours $GPU
python benchmark_synthetic.py separability scar-km2 $GPU

python benchmark_synthetic.py group_gap supervised $GPU
python benchmark_synthetic.py group_gap negative $GPU
python benchmark_synthetic.py group_gap sar-em $GPU
python benchmark_synthetic.py group_gap ours $GPU
python benchmark_synthetic.py group_gap scar-km2 $GPU

python benchmark_synthetic_robustness.py ours $GPU
