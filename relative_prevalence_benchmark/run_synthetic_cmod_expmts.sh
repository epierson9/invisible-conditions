 #!/bin/bash

python benchmark_simulated_civilcomments.py cmod ours gender 2
python benchmark_simulated_civilcomments.py cmod observed gender 2
python benchmark_simulated_civilcomments.py cmod true gender 2

python benchmark_simulated_civilcomments.py cmod ours religion 2
python benchmark_simulated_civilcomments.py cmod observed religion 2
python benchmark_simulated_civilcomments.py cmod true religion 2

python benchmark_simulated_civilcomments.py cmod ours identity 2
python benchmark_simulated_civilcomments.py cmod observed identity 2
python benchmark_simulated_civilcomments.py cmod true identity 2

python benchmark_simulated_civilcomments.py cmod ours sexual_orientation 2
python benchmark_simulated_civilcomments.py cmod observed sexual_orientation 2
python benchmark_simulated_civilcomments.py cmod true sexual_orientation 2
