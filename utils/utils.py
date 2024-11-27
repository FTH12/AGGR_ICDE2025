import csv
import os
import random
import time
from pathlib import Path, PurePath
from typing import Tuple

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, SpectralClustering

def sk_clustering(X: torch.Tensor, n_clusters: int, name: str = "kmeans") -> np.ndarray:
    if name == "kmeans":
        model = KMeans(n_clusters=n_clusters)
        label_pred = model.fit(X).labels_
        return label_pred
    if name == "spectral":
        model = SpectralClustering(n_clusters=n_clusters, affinity="precomputed")
        label_pred = model.fit(X).labels_
        return label_pred
    raise NotImplementedError

def make_parent_dirs(target_path: PurePath) -> None:
    if not target_path.parent.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)

def refresh_file(target_path: str = None) -> None:
    if target_path is not None:
        target_path: PurePath = Path(target_path)
        if target_path.exists():
            target_path.unlink()
        make_parent_dirs(target_path)
        target_path.touch()

def save_model(model_filename: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, current_epoch: int, loss: float) -> None:
    model_path = get_modelfile_path(model_filename)
    torch.save(
        {
            "epoch": current_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        model_path,
    )

def load_model(model_filename: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, device: torch.device = torch.device("cpu")) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
    model_path = get_modelfile_path(model_filename)
    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    return model, optimizer, epoch, loss

def get_modelfile_path(model_filename: str) -> PurePath:
    model_path: PurePath = Path(f".checkpoints/{model_filename}.pt")
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
    return model_path

def check_modelfile_exists(model_filename: str) -> bool:
    return get_modelfile_path(model_filename).exists()

def get_str_time():
    return "time_" + time.strftime("%m%d%H%M%S", time.localtime())

def node_homo(adj: sp.spmatrix, labels: torch.Tensor) -> float:
    adj_coo = adj.tocoo()
    adj_coo.data = ((labels[adj_coo.col] == labels[adj_coo.row]).cpu().numpy().astype(int))
    return (np.asarray(adj_coo.sum(1)).flatten() / np.asarray(adj.sum(1)).flatten()).mean()

def edge_homo(adj: sp.spmatrix, labels: torch.Tensor) -> float:
    return ((labels[adj.tocoo().col] == labels[adj.tocoo().row]).cpu().numpy() * adj.data).sum() / adj.sum()

def eliminate_zeros(adj: sp.spmatrix) -> sp.spmatrix:
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    return adj

def csv2file(target_path: str, thead: Tuple[str] = None, tbody: Tuple = None, refresh: bool = False, is_dict: bool = False) -> None:
    target_path: PurePath = Path(target_path)
    if refresh:
        refresh_file(target_path)

    make_parent_dirs(target_path)

    with open(target_path, "a+", newline="", encoding="utf-8") as csvfile:
        csv_write = csv.writer(csvfile)
        if os.stat(target_path).st_size == 0 and thead is not None:
            csv_write.writerow(thead)
        if tbody is not None:
            if is_dict:
                dict_writer = csv.DictWriter(csvfile, fieldnames=tbody[0].keys())
                for elem in tbody:
                    dict_writer.writerow(elem)
            else:
                csv_write.writerow(tbody)

def set_seed(seed: int = 4096) -> None:
    if seed is not False:
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def set_device(gpu: str = "0") -> torch.device:
    max_device = torch.cuda.device_count() - 1
    if gpu == "none":
        print("Use CPU.")
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        if not gpu.isnumeric():
            raise ValueError(f"args.gpu:{gpu} invalid")
        if int(gpu) <= max_device:
            print(f"use cuda:{gpu}。")
            device = torch.device(f"cuda:{gpu}")
            torch.cuda.set_device(device)
        else:
            print(f"use cpu")
            device = torch.device("cpu")
    else:
        print("use cpu")
        device = torch.device("cpu")
    return device

def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.Tensor:
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

def torch_sparse_to_dgl_graph(torch_sparse_mx):
    torch_sparse_mx = torch_sparse_mx.coalesce()
    indices = torch_sparse_mx.indices()
    values = torch_sparse_mx.values()
    rows_, cols_ = indices[0, :], indices[1, :]
    dgl_graph = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0])
    dgl_graph.edata["w"] = values.detach()
    return dgl_graph
