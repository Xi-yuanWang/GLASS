import subprocess
import time

# pretrain GNN to produce embedding
def emb(dg):
    dataset, gpu_id = dg
    cmd = f"nohup python GNNEmb.py --use_{feature} --device {gpu_id} --dataset {dataset} --name {dataset} >  Emb/{dataset}{gpu_id} 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)

# test GLASS
def test(dg):
    dataset, gpu_id = dg
    cmd = f"nohup python GLASSTest.py --use_{feature} --use_seed --use_maxzeroone --repeat 10 --device {gpu_id} --dataset {dataset} >  out/{dataset}{gpu_id}.test3 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)

# ablation test
def gnn(dg):
    dataset, gpu_id = dg
    cmd = f"nohup python GLASSTest.py  --use_{feature} --use_seed --repeat 10 --device {gpu_id} --dataset {dataset} >  out/{dataset}{gpu_id}.gnn 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)



# dg is pair of dataset name and GPU id.
feature = "nodeid"
real = [("ppi_bp", 0), ("hpo_metab", 0), ("hpo_neuro", 0), ("em_user", 0)]
for dg in real:
    gnn(dg)
feature = "one"
syn = [("density", 0), ("component", 0),("coreness", 0),("cut_ratio", 0)] 
for dg in syn:
    gnn(dg)


