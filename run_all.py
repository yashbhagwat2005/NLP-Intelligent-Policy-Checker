"""Run the whole mid-term pipeline in order:  python run_all.py"""
import os
import subprocess
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
STEPS = ["prepare_data.py", "eda.py", "train_models.py", "compliance_checker.py"]

for step in STEPS:
    print(f"\n{'=' * 70}\n RUNNING {step}\n{'=' * 70}")
    subprocess.run([sys.executable, os.path.join(SRC, step)], check=True, cwd=SRC)
