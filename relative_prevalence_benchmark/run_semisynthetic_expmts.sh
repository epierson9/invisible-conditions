 
 #!/bin/bash
GPU_ARG=1
#python benchmark_semisynthetic.py random supervised 1
#python benchmark_semisynthetic.py random scar-km2 1
#python benchmark_semisynthetic.py random sar-em 1
#python benchmark_semisynthetic.py random negative $GPU_ARG 
#python benchmark_semisynthetic.py random ours 1

#python benchmark_semisynthetic.py ipv supervised $GPU_ARG
#python benchmark_semisynthetic.py ipv scar-km2 1
#python benchmark_semisynthetic.py ipv sar-em 1
#python benchmark_semisynthetic.py ipv negative $GPU_ARG
#python benchmark_semisynthetic.py ipv ours 1

#python benchmark_semisynthetic.py corr supervised 1
#python benchmark_semisynthetic.py corr scar-km2 1
#python benchmark_semisynthetic.py corr sar-em 1
#python benchmark_semisynthetic.py corr negative $GPU_ARG
#python benchmark_semisynthetic.py corr ours 2

#python benchmark_semisynthetic.py high_rp supervised $GPU_ARG
#python benchmark_semisynthetic.py high_rp negative $GPU_ARG
#python benchmark_semisynthetic.py high_rp ours $GPU_ARG
python benchmark_semisynthetic.py high_rp sar-em $GPU_ARG
#python benchmark_semisynthetic.py high_rp scar-km2 $GPU_ARG
