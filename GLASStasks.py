import subprocess


# pretrain GNN to produce embedding
def emb(dataset, gpu_id):
    cmd = f"nohup python GNNEmb.py --use_{feature} --device {gpu_id} --dataset {dataset} --name {dataset} >  Emb/{dataset}{gpu_id} 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


# test GLASS
def test(dataset, gpu_id):
    cmd = f"nohup python GLASSTest.py --use_{feature} --use_seed --use_maxzeroone --repeat 10 --device {gpu_id} --dataset {dataset} >  out/{dataset}.test 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


# ablation analysis using plain GNN
def gnn(dataset, gpu_id):
    cmd = f"nohup python GLASSTest.py  --use_{feature} --use_seed --repeat 10 --device {gpu_id} --dataset {dataset} >  out/{dataset}.gnn 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


# dg is pair of dataset name and GPU id.
# reproduce results in GLASS paper.

feature = "nodeid"
real = [("ppi_bp", 0), ("hpo_metab", 1), ("hpo_neuro", 2), ("em_user", 3)]
for dg in real:
    test(*dg)
'''
feature = "one"
syn = [("density", 0), ("component", 1), ("coreness", 2), ("cut_ratio", 3)]
for dg in syn:
    test(*dg)
'''