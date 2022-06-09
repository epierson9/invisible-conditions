 #!/bin/bash
GPU_ARG=1
python benchmark_civilcomments.py cmod ours gender $GPU_ARG
python benchmark_civilcomments.py cmod observed gender $GPU_ARG

python benchmark_civilcomments.py cmod ours religion $GPU_ARG
python benchmark_civilcomments.py cmod observed religion $GPU_ARG

python benchmark_civilcomments.py cmod ours identity $GPU_ARG
python benchmark_civilcomments.py cmod observed identity $GPU_ARG

python benchmark_civilcomments.py cmod ours sexual_orientation $GPU_ARG
python benchmark_civilcomments.py cmod observed sexual_orientation $GPU_ARG

python benchmark_civilcomments.py cmod_higher_thresh ours gender $GPU_ARG
python benchmark_civilcomments.py cmod_higher_thresh observed gender $GPU_ARG

python benchmark_civilcomments.py cmod_higher_thresh ours religion $GPU_ARG
python benchmark_civilcomments.py cmod_higher_thresh observed religion $GPU_ARG

python benchmark_civilcomments.py cmod_higher_thresh ours identity $GPU_ARG
python benchmark_civilcomments.py cmod_higher_thresh observed identity $GPU_ARG

python benchmark_civilcomments.py cmod_higher_thresh ours sexual_orientation $GPU_ARG
python benchmark_civilcomments.py cmod_higher_thresh observed sexual_orientation $GPU_ARG


python benchmark_civilcomments.py cmod_highest_thresh ours gender $GPU_ARG
python benchmark_civilcomments.py cmod_highest_thresh observed gender $GPU_ARG

python benchmark_civilcomments.py cmod_highest_thresh ours religion $GPU_ARG
python benchmark_civilcomments.py cmod_highest_thresh observed religion $GPU_ARG

python benchmark_civilcomments.py cmod_highest_thresh ours identity $GPU_ARG
python benchmark_civilcomments.py cmod_highest_thresh observed identity $GPU_ARG

python benchmark_civilcomments.py cmod_highest_thresh ours sexual_orientation $GPU_ARG
python benchmark_civilcomments.py cmod_highest_thresh observed sexual_orientation $GPU_ARG
