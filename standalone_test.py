import subprocess
import time
from multiprocessing import Process, Pool
import random

feature = "one"

def work(dg):
    dataset, gpu_id = dg
    cmd = f"nohup python standalone.py  --use_{feature} --repeat 1 --device {gpu_id} --dataset {dataset} --path standalone/ --name {dataset} > standalone/{dataset}{gpu_id} 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)

def test(dg):
    dataset, gpu_id = dg
    cmd = f"nohup python standalone.py --test --use_{feature} --repeat 10 --use_seed --device {gpu_id} --dataset {dataset} --path standalone/ --name {dataset} > standalone/{dataset}{gpu_id}.test 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)

for dg in [("hpo_neuro", 0), ("hpo_metab", 1), ("ppi_bp", 2), ("em_user", 3)]:
    work(dg)


for dg in [("density", 0), ("cut_ratio", 1), ("coreness",2)]:
    test(dg)
