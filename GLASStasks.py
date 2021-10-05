import subprocess
import time


def test(dg):
    dataset, gpu_id = dg
    cmd = f"nohup python DistGNNTest.py --use_{feature} --use_seed --use_maxzeroone --repeat 10 --device {gpu_id} --dataset {dataset} >  out/{dataset}{gpu_id}.test2 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def abl(dg):
    dataset, gpu_id = dg
    cmd = f"nohup python DistGNNTest.py  --use_{feature} --use_seed --repeat 10 --device {gpu_id} --dataset {dataset} >  out/{dataset}{gpu_id}.abl 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


# Synthetic datasets
feature = "one"
# dg is pair of dataset name and GPU id.
syn = [("coreness", 0), ("cut_ratio", 0), ("density", 0), ("component", 0)]
for dg in syn:
    test(dg)

# Real-world datasets
feature = "nodeid"
real = [("ppi_bp", 0), ("hpo_metab", 0), ("hpo_neuro", 0), ("em_user", 0)]
for dg in syn:
    test(dg)
