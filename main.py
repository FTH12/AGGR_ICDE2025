# pylint: disable=line-too-long,invalid-name,
import argparse
import random
import torch
from graph_datasets import load_data

from models import AGGR_batch
from models.AGGR import AGGR
from utils import check_modelfile_exists
from utils import csv2file
from utils import evaluation
from utils import get_str_time
from utils import set_device


def parse_args():
    parser = argparse.ArgumentParser(
        prog="AGGR",  # 程序名称
        description="Adaptive Granular Graph Rewiring via Granular-ball for Graph Clustering",  # 程序描述
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='Cora'
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="gpu id",
    )
    parser.add_argument(
        "--sim",
        type=str,
        default='dot',
        help="sim method: dot cos per",
    )
    parser.add_argument(
        "--quilty",
        type=str,
        default='homo',
        help="quilty method: detach homo edges deg",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--pre",
        type=int,
        default=150
    )
    parser.add_argument(
        "--train",
        type=int,
        default=50
    )
    parser.add_argument(
        "--add",
        type=float,
        default=0.01
    )
    parser.add_argument(
        "--delr",
        type=float,
        default=0.02
    )
    args = parser.parse_args()
    return args



if __name__ == '__main__':

    n_gnn_layers = {
        "Cora": 8,
        "Citeseer": 3,
        "ACM": 3,
        "Pubmed": 35,
        "BlogCatalog":1,
        "Flickr": 1,
        "Reddit":3,
    }

    inner_act = {
        "Cora": torch.tanh,
        "Citeseer": torch.sigmoid,
        "ACM": lambda x: x,
        "Pubmed": lambda x: x,
        "BlogCatalog": lambda x: x,
        "Flickr": lambda x: x,
        "Squirrel": lambda x: x,
        "Reddit": lambda x: x,
    }
    # 更新图结构的周期
    udp = {
        "Cora": 5,
        "Citeseer": 40,
        "ACM": 40,
        "Pubmed": 10,
        "BlogCatalog": 20,
        "Flickr": 40,
        "Squirrel": 40,
        "Reddit": 40,
    }
    gr_epochs = {
        "Cora": 4,
        "Citeseer": 4,
        "ACM": 8,
        "Pubmed": 2,
        "BlogCatalog": 5,
        "Flickr": 5,
        "Reddit": 1,
    }
    regularization = {
        "Cora": 1,
        "Citeseer": 2 ,
        "ACM": 0,
        "Pubmed": 0,
        "BlogCatalog": 0,
        "Flickr": 1,
        "Reddit": 0,
    }
    source = {
        "Cora": "dgl",
        "Citeseer": "dgl",
        "ACM": "sdcn",
        "Pubmed": "dgl",
        "BlogCatalog": "cola",
        "Flickr": "cola",
        "Reddit": "dgl",
    }
    args = parse_args()
    dataset_name = args.dataset
    final_params = {}
    dim = 500
    n_lin_layers = 1
    dump = True
    device = set_device(str(args.gpu_id))
    if dataset_name == "Reddit":
        aggr = AGGR_batch
    else:
        aggr = AGGR
    final_params["lr"] = args.lr
    final_params["pre_epochs"] = args.pre
    final_params["epochs"] = args.train
    final_params['sim'] = args.sim
    final_params['quilty'] = args.quilty
    final_params["add_ratio"] = args.add
    final_params["del_ratio"] = args.delr
    final_params["dataset"] = dataset_name
    time_name = get_str_time()
    save_file = f"results/AGGR/AGGR_{dataset_name}_gnn_{n_gnn_layers[dataset_name]}_{time_name[:9]}.csv"
    graph, labels, n_clusters = load_data(
        dataset_name=dataset_name,
        source=source[dataset_name],
        verbosity=2,
    )
    features = graph.ndata["feat"]
    if dataset_name in ("Cora", "Pubmed"):
        graph.ndata["feat"][(features - 0.0) > 0.0] = 1.0
    adj_csr = graph.adj_external(scipy_fmt="csr")
    adj_sum_raw = adj_csr.sum()
    warmup_filename = f"AGGR_{dataset_name}_run_gnn_{n_gnn_layers[dataset_name]}"
    if not check_modelfile_exists(warmup_filename):
        print("warmup first")
        model = aggr(
            hidden_units=[dim],
            in_feats=features.shape[1],
            n_clusters=n_clusters,
            n_gnn_layers=n_gnn_layers[dataset_name],
            n_lin_layers=n_lin_layers,
            lr=args.lr,
            n_pretrain_epochs=args.pre,
            n_epochs=args.train,
            norm="sym",
            renorm=True,
            warmup_filename=warmup_filename,
            inner_act=inner_act[dataset_name],
            udp=udp[dataset_name],
            regularization=regularization[dataset_name],
        )
        model.fit(
            graph=graph,
            device=device,
            node_ratio=args.add,
            del_edge_ratio=args.delr,
            gr_epochs=0,
            labels=labels,
            adj_sum_raw=adj_sum_raw,
            load=False,
            dump=dump,
        )
    runs = 1
    seed_list = [
        random.randint(0, 999999) for _ in range(runs)
    ]
    for run_id in range(runs):
        final_params["run_id"] = run_id
        seed = seed_list[run_id]
        final_params["seed"] = seed
        reset = dataset_name == "Citeseer"

        model = aggr(
            hidden_units=[dim],
            in_feats=features.shape[1],
            n_clusters=n_clusters,
            n_gnn_layers=n_gnn_layers[dataset_name],
            n_lin_layers=n_lin_layers,
            lr=args.lr,
            n_pretrain_epochs=args.pre,
            n_epochs=args.train,
            norm="sym",
            renorm=True,
            warmup_filename=warmup_filename,
            inner_act=inner_act[dataset_name],
            udp=udp[dataset_name],
            reset=reset,
            regularization=regularization[dataset_name],
            seed=seed,
        )

        model.fit(
            graph=graph,
            device=device,
            node_ratio=args.add,
            del_edge_ratio=args.delr,
            gr_epochs=gr_epochs[dataset_name],
            labels=labels,
            adj_sum_raw=adj_sum_raw,
            load=True,
            dump=dump,
            simm=args.sim,
            quilty=args.quilty
        )

        with torch.no_grad():
            z_detached = model.get_embedding()
            Q = model.get_Q(z_detached)
            q = Q.detach().data.cpu().numpy().argmax(1)
        (
            ARI_score,
            NMI_score,
            AMI_score,
            ACC_score,
            Micro_F1_score,
            Macro_F1_score,
            purity,  # 纯度
        ) = evaluation(labels, q)

        # 打印评估结果
        print("\n"
              f"ARI:{ARI_score}\n"
              f"NMI:{NMI_score}\n"
              f"AMI:{AMI_score}\n"
              f"ACC:{ACC_score}\n"
              f"Micro F1:{Micro_F1_score}\n"
              f"Macro F1:{Macro_F1_score}\n"
              f"purity_score:{purity}\n")
        final_params["qARI"] = ARI_score
        final_params["qNMI"] = NMI_score
        final_params["qACC"] = ACC_score
        if save_file is not None:
            csv2file(
                target_path=save_file,
                thead=list(final_params.keys()),
                tbody=list(final_params.values()),
            )
            print(f"write to {save_file}")

        print(final_params)