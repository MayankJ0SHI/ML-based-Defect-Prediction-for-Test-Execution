#!/usr/bin/env python
# check_and_push.py
import os
import subprocess
import time

# Access environment variables
GH_USERNAME = os.environ.get('GH_USERNAME')
GH_PAT = os.environ.get('GH_PAT')

if GH_USERNAME and GH_PAT:
    print(f'PYTHON GH_USERNAME: {GH_USERNAME}, GH_PAT: {GH_PAT}')
else:
    print("Not enough environment variables provided.")
    exit(1)

while True:
    # Sleep for 24 hours
    time.sleep(20)

    # Run the push script with environment variables
    subprocess.run(["/usr/src/app/push_to_github.sh", GH_USERNAME, GH_PAT])
