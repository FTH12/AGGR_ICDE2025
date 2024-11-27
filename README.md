# AGGR_ICDE2025
Adaptive Granular Graph Rewiring via Granular-ball for Graph Clustering-ICDE2025

## Installation
- See [requirements.txt](./requirements.txt)


```bash
conda install --yes --file requirements.txt
```
## Usage
- See [run.sh](./run.sh)

```bash
# Activate the env, e.g., for linux run:
$ conda activate AGGR

# Run AGGR
$ python main.py --dataset Cora --gpu_id 0 --lr 0.001 --pre 150 --train 50 --add 0.01 --delr 0.02
```
