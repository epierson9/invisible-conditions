 #!/bin/bash
GPU_ARG=3
python benchmark_mimic.py ipv ours ethnicity $GPU_ARG 
python benchmark_mimic.py ipv ours marital_status $GPU_ARG
python benchmark_mimic.py ipv ours insurance $GPU_ARG

python benchmark_mimic.py ipv negative ethnicity $GPU_ARG
python benchmark_mimic.py ipv negative insurance $GPU_ARG
python benchmark_mimic.py ipv negative marital_status $GPU_ARG

python benchmark_mimic.py ipv observed ethnicity $GPU_ARG
python benchmark_mimic.py ipv observed insurance $GPU_ARG
python benchmark_mimic.py ipv observed marital_status $GPU_ARG
