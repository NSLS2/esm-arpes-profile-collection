import os
import sys
import subprocess

cmd = "ps aux | grep -i '[p]ixi run' | grep -i 'collection' | grep -i 'start' | wc -l"
max_instances = 5
result = subprocess.run(
    cmd,
    shell=True,
    capture_output=True,
    text=True
)
num_instances = int(result.stdout.strip()) - 1

assert num_instances <= max_instances, "Max number of instances exceeded"

print(str(num_instances) + " instances of bsui currently running")
