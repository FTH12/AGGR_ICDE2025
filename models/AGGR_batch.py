"""
Adaptive Granular Graph Rewiring via Granular-ball for graph clustering
"""
import copy
import gc
import os
import pickle
import random
import time
from typing import Callable, List

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from torch import nn
from modules.granular import Granular
from modules import InnerProductDecoder, LinTrans, preprocess_graph, SampleDecoder
from utils import eliminate_zeros, set_seed, sparse_mx_to_torch_sparse_tensor, torch_sparse_to_dgl_graph

def save_GB_data(file_path, GB_node_list, GB_graph_list, GB_center_list):
    data = {
        "GB_node_list": GB_node_list,
        "GB_graph_list": GB_graph_list,
        "GB_center_list": GB_center_list
    }
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved GB data to {file_path}")
def load_GB_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded GB data from {file_path}")
    return data["GB_node_list"], data["GB_graph_list"], data["GB_center_list"]


class GBHoLe_batch_Reddit(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_units: List[int],
        n_clusters: int,
        n_lin_layers: int = 1,
        n_gnn_layers: int = 10,
        n_cls_layers: int = 1,
        lr: float = 0.001,
        n_epochs: int = 400,
        n_pretrain_epochs: int = 400,
        norm: str = "sym",
        renorm: bool = True,
        warmup_filename: str = "AGGR_warmup",
        inner_act: Callable = lambda x: x,
        udp: int = 10,
        reset: bool = False,
        regularization: float = 0,
        seed: int = 4096,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_gnn_layers = n_gnn_layers
        self.n_lin_layers = n_lin_layers
        self.n_cls_layers = n_cls_layers
        self.hidden_units = hidden_units
        self.lr = lr
        self.n_epochs = n_epochs
        self.n_pretrain_epochs = n_pretrain_epochs
        self.norm = norm
        self.renorm = renorm
        self.device = None
        self.sm_fea_s = None
        self.adj_label = None
        self.lbls = None
        self.warmup_filename = warmup_filename
        self.udp = udp
        self.reset = reset
        self.labels = None
        self.adj_sum_raw = None
        self.adj_orig = None
        self.regularization = regularization
        set_seed(seed)
        self.dims = [in_feats] + hidden_units

        self.encoder = LinTrans(self.n_lin_layers, self.dims)
        self.cluster_layer = nn.Parameter(torch.Tensor(self.n_clusters, hidden_units[-1]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.sample_decoder = SampleDecoder(act=lambda x: x)
        self.inner_product_decoder = InnerProductDecoder(act=inner_act)

        self.best_model = copy.deepcopy(self)


    def reset_weights(self):
        self.encoder = LinTrans(self.n_lin_layers, self.dims)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def bce_loss(preds, labels, norm=1.0, pos_weight=None):

        return norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    def aux_objective(self, z_detached):

        sample_size = int(self.n_nodes * 1)
        s = random.sample(range(0, self.n_nodes), sample_size)
        out = z_detached[s, :].float()
        C = self.on_labels[s,:].float()

        t1 = torch.matmul(torch.t(C), C)
        t1 = torch.matmul(t1, t1)
        t1 = torch.trace(t1)

        t2 = torch.matmul(torch.t(out), out)
        t2 = torch.matmul(t2, t2)
        t2 = torch.trace(t2)

        t3 = torch.matmul(torch.t(out), C)
        t3 = torch.matmul(t3, torch.t(t3))
        t3 = torch.trace(t3)

        aux_objective_loss = 1 / (sample_size ** 2) * (t1 + t2 - 2 * t3)

        return aux_objective_loss

    @staticmethod
    def get_fd_loss(z, norm=1):
        norm_ff = z / (z**2).sum(0, keepdim=True).sqrt()
        coef_mat = torch.mm(norm_ff.t(), norm_ff)
        coef_mat.div_(2.0)
        a = torch.arange(coef_mat.size(0), device=coef_mat.device)
        L_fd = norm * F.cross_entropy(coef_mat, a)
        return L_fd

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def get_cluster_center(self, z=None):

        if z is None:
            z = self.get_embedding()

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        _ = kmeans.fit_predict(z.data.cpu().numpy())
        self.cluster_layer.data = torch.Tensor(kmeans.cluster_centers_).to(self.device)

    def get_Q(self, z):

        q = 1.0 / (1.0 + torch.sum(torch.pow(z.cpu().unsqueeze(1) - self.cluster_layer.cpu(), 2), 2) / 1)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


    def _re_pretrain(self):

        best_loss = 1e9

        for epoch in range(self.n_pretrain_epochs):
            self.train()
            self.optimizer.zero_grad()
            t = time.time()

            z = self.encoder(self.sm_fea_s)

            index = list(range(len(z)))
            split_list = data_split(index, 10000)

            ins_loss = 0

            for batch in split_list:
                z_batch = z[batch]
                A_batch = torch.FloatTensor(self.lbls[batch, :][:, batch].toarray()).to(self.device)
                preds = self.inner_product_decoder(z_batch).view(-1)

                pos_weight_b = torch.FloatTensor([float(A_batch.shape[0] * A_batch.shape[0] - A_batch.sum()) / A_batch.sum()]).to(self.device)
                norm_weights_b = (A_batch.shape[0] * A_batch.shape[0] / float((A_batch.shape[0] * A_batch.shape[0] - A_batch.sum()) * 2))
                loss = self.bce_loss(preds, A_batch.view(-1), norm_weights_b, pos_weight_b)

                ins_loss += loss.item()

                (loss).backward(retain_graph=True)
                self.optimizer.step()

            torch.cuda.empty_cache()
            gc.collect()

            print(f"Cluster Epoch: {epoch}, ins_loss={ins_loss}, time={time.time() - t:.5f}")

            if ins_loss < best_loss:
                del self.best_model
                self.best_model = copy.deepcopy(self.to(self.device))


    def _re_ce_train(self):

        best_loss = 1e9

        self.get_cluster_center()

        for epoch in range(self.n_epochs):
            self.train()

            if epoch % self.udp == 0 or epoch == self.n_epochs - 1:
                import torch
                import time
                import gc
                import numpy as np
                from scipy.sparse import csr_matrix
                from torch.nn.functional import kl_div, binary_cross_entropy_with_logits as bce_loss
                with torch.no_grad():
                    z_detached = self.get_embedding()
                    Q = self.get_Q(z_detached)
                    q = Q.detach().data.cpu().numpy().argmax(1)

            self.optimizer.zero_grad()
            t = time.time()

            z = self.encoder(self.sm_fea_s)

            index = list(range(len(z)))
            split_list = data_split(index, 10000)

            ins_loss = 0

            for batch in split_list:
                z_batch = z[batch]

                sub_adj_matrix = self.lbls[batch, :][:, batch]

                indices = torch.LongTensor(sub_adj_matrix.nonzero())
                values = torch.FloatTensor(sub_adj_matrix.data)
                shape = sub_adj_matrix.shape
                A_batch = torch.sparse_coo_tensor(indices, values, shape).to(self.device)
                A_batch = A_batch.to_dense()
                preds = self.inner_product_decoder(z_batch).view(-1)

                q = self.get_Q(z_batch)
                p = self.target_distribution(Q[batch].detach())
                kl_loss = F.kl_div(q.log(), p, reduction="batchmean")

                pos_weight_b = torch.FloatTensor(
                    [(float(A_batch.shape[0] * A_batch.shape[0] - A_batch.sum()) / A_batch.sum())]).to(self.device)
                norm_weights_b = (A_batch.shape[0] * A_batch.shape[0] / float(
                    (A_batch.shape[0] * A_batch.shape[0] - A_batch.sum()) * 2))
                loss = self.bce_loss(preds, A_batch.view(-1), norm_weights_b, pos_weight_b)
                ins_loss += loss.item()
                aux_loss = self.aux_objective(z)
                self.optimizer.zero_grad()
                (aux_loss + loss + kl_loss).backward(retain_graph=True)
                self.optimizer.step()

            print(f"Cluster Epoch: {epoch}, ins_loss={ins_loss}, kl_loss={kl_loss.item()},aux_loss={aux_loss}, time={time.time() - t:.5f}")

            if ins_loss < best_loss:
                best_loss = ins_loss
                del self.best_model
                self.best_model = copy.deepcopy(self.to(self.device))

            torch.cuda.empty_cache()
            gc.collect()

    def compute_similarity_in_batches(self, z, batch_size=1024):
        n_nodes = z.shape[0]
        sims = []

        for i in range(0, n_nodes, batch_size):
            for j in range(0, n_nodes, batch_size):
                start_i, end_i = i, min(i + batch_size, n_nodes)
                start_j, end_j = j, min(j + batch_size, n_nodes)
                sim = torch.mm(z[start_i:end_i], z[start_j:end_j].T)
                sims.append(sim)

        return torch.block_diag(*sims)
    def get_pseudo_label(self, node_rate=0.2):
        with torch.no_grad():
            z_detached = self.get_embedding()
            Q = self.get_Q(z_detached)
            soft = Q.detach()

        hard = soft.argmax(dim=1).view(-1).cpu().numpy()

        hard_mask = np.array([False for _ in range(len(hard))], dtype=np.bool_)
        for c in range(self.n_clusters):
            add_num = int(node_rate * soft.shape[0])

            c_col = soft[:, c].detach().cpu().numpy()
            c_col_idx = c_col.argsort()[::-1][:add_num]
            top_c_idx = c_col_idx
            hard[top_c_idx] = c

            hard_mask[top_c_idx] = True

            print(f"class {c}, num={len(top_c_idx)}")

        hard[~hard_mask] = -1
        self.pseudo_label = hard
        self.hard_mask = hard_mask

    def _train(self):
        self._re_ce_train()

    def _pretrain(self):
        self._re_pretrain()

    def add_edge(self, edges, label, n_nodes, add_edge_rate, z):
        u, v = edges[0].numpy(), edges[1].numpy()
        lbl_np = label
        final_u, final_v = [], []

        for cluster_label in np.unique(lbl_np):
            if cluster_label == -1:
                continue

            print(f"************ label == {cluster_label} ************")
            same_class_nodes = np.where(lbl_np == cluster_label)[0]
            n_cluster_nodes = len(same_class_nodes)

            if n_cluster_nodes <= 1:
                continue

            add_num = int(add_edge_rate * n_cluster_nodes ** 2)

            cluster_embeds = z[same_class_nodes]
            sims = []

            chunk_size = 1024
            for start in range(0, n_cluster_nodes, chunk_size):
                end = min(start + chunk_size, n_cluster_nodes)
                sim_chunk = torch.mm(cluster_embeds, cluster_embeds[start:end].T)
                sims.append(sim_chunk)

            sims = torch.cat(sims, dim=1)
            sims.fill_diagonal_(-float("inf"))

            flat_indices = torch.topk(sims.view(-1), min(add_num, sims.numel())).indices
            row_indices = flat_indices // n_cluster_nodes
            col_indices = flat_indices % n_cluster_nodes

            new_u = same_class_nodes[row_indices.cpu().numpy()]
            new_v = same_class_nodes[col_indices.cpu().numpy()]

            final_u.extend(new_u)
            final_v.extend(new_v)

        new_rows = np.concatenate([u, final_u, final_v])
        new_cols = np.concatenate([v, final_v, final_u])
        new_data = np.ones(len(new_rows))
        adj_csr = sp.csr_matrix((new_data, (new_rows, new_cols)), shape=(n_nodes, n_nodes))

        adj_csr[adj_csr > 1] = 1
        adj_csr = eliminate_zeros(adj_csr)
        print(f"after add edge, final edges num={adj_csr.sum()}")
        return adj_csr

    def del_edge(self, edges, label, n_nodes, del_edge_rate=0.25):
        u, v = edges[0].numpy(), edges[1].numpy()
        lbl_np = label

        inter_class_bool = (lbl_np[u] != lbl_np[v]) & (lbl_np[u] != -1) & (lbl_np[v] != -1)
        inter_class_idx = np.where(inter_class_bool)[0]
        inter_class_edge_len = len(inter_class_idx)

        remain_edge_len = max(0, inter_class_edge_len - int(inter_class_edge_len * del_edge_rate))
        retained_inter_class_idx = np.random.choice(inter_class_idx, remain_edge_len, replace=False)

        retained_edges_idx = np.setdiff1d(np.arange(len(u)), inter_class_idx)
        final_edges_idx = np.concatenate([retained_edges_idx, retained_inter_class_idx])

        new_u = u[final_edges_idx]
        new_v = v[final_edges_idx]

        row = np.concatenate([new_u, new_v])
        col = np.concatenate([new_v, new_u])
        data = np.ones(len(row))
        adj_csr = sp.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))

        adj_csr[adj_csr > 1] = 1
        adj_csr = eliminate_zeros(adj_csr)
        print(f"after del edge, final edges num={adj_csr.sum()}")
        return adj_csr

    def update_features(self, adj, first=False):
        sm_fea_s = sp.csr_matrix(self.features).toarray()

        adj_cp = copy.deepcopy(adj)
        self.adj_label = adj_cp

        adj_norm_s = preprocess_graph(
            adj_cp,
            self.n_gnn_layers,
            norm=self.norm,
            renorm=self.renorm,
        )

        adj_csr = adj_norm_s[0] if len(adj_norm_s) > 0 else adj_cp

        print("Laplacian Smoothing...")
        for a in adj_norm_s:
            sm_fea_s = a.dot(sm_fea_s)
        self.sm_fea_s = torch.FloatTensor(sm_fea_s).to(self.device)

        self.pos_weight = torch.FloatTensor([
            (float(adj_csr.shape[0] * adj_csr.shape[0] - adj_csr.sum()) /
             adj_csr.sum())
        ]).to(self.device)

        self.norm_weights = (adj_csr.shape[0] * adj_csr.shape[0] / float(
            (adj_csr.shape[0] * adj_csr.shape[0] - adj_csr.sum()) * 2))

        self.lbls =adj_csr

    def fit(
        self,
        graph: dgl.DGLGraph,
        device: torch.device,
        del_edge_ratio=0.1,
        node_ratio=0.2,
        gr_epochs=10,
        labels=None,
        adj_sum_raw=None,
        load=False,
        dump=True,
        simm = 'dot',
        quilty = 'detach'
    ):
        self.device = device
        self.features = graph.ndata["feat"]
        adj = self.adj_orig = graph.adj_external(scipy_fmt="csr")
        self.n_nodes = self.features.shape[0]
        self.labels = labels
        self.gb_layer = Granular(quity=quilty, sim=simm)
        self.adj_sum_raw = adj_sum_raw

        adj = eliminate_zeros(adj)

        self.to(self.device)

        self.update_features(adj)

        adj = self.adj_label

        from utils.utils import check_modelfile_exists
        if load and check_modelfile_exists(self.warmup_filename):
            from utils.utils import load_model
            self, self.optimizer, _, _ = load_model(self.warmup_filename, self, self.optimizer, self.device)
            self.to(self.device)
            print(f"model loaded from {self.warmup_filename} to {self.device}")
        else:
            self._pretrain()
            if dump:
                from utils.utils import save_model
                save_model(self.warmup_filename, self, self.optimizer, None, None)

        adj_pre = copy.deepcopy(adj)
        self.on_labels = F.one_hot(self.labels, num_classes=self.n_clusters).to(self.device)
        for gr_ep in range(gr_epochs):
            torch.cuda.empty_cache()
            gc.collect()

            print(f"==============GR epoch:{gr_ep} ===========")
            # with torch.no_grad():
            #     z_detached = self.get_embedding()
            adj_new = eliminate_zeros(
                self.update_adj(
                    adj_pre,
                    ratio=node_ratio,
                    del_ratio=del_edge_ratio,
                ))

            self.update_features(adj=adj_new)
            self._train()

            if self.reset:
                self.reset_weights()
                self.to(self.device)


    def update_adj(
        self,
        adj,
        ratio=0.2,
        del_ratio=0.005,):
        import torch.sparse

        coo = adj.tocoo()
        indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        shape = coo.shape
        adj_tensor = torch.sparse_coo_tensor(indices, values, size=shape).to(self.device)

        z_detached = self.get_embedding()

        self.gb_layer.z_detached = z_detached

        print('Generating GB')
        adj_csr = sp.csr_matrix((adj_tensor.coalesce().values().cpu().numpy(),
                                    (adj_tensor.coalesce().indices()[0].cpu().numpy(),
                                    adj_tensor.coalesce().indices()[1].cpu().numpy())),
                                shape=adj_tensor.shape)
        GB_node_list, GB_graph_list, GB_center_list = self.gb_layer.forward(adj_csr)
        print('GB success generate')
        adj_new = self.modify_adj(adj_tensor, GB_node_list, GB_center_list, z_detached, ratio, del_ratio)
        return adj_new


    def batched_topk(self,similarity_matrix, del_num, batch_size):
        M, N = similarity_matrix.shape
        total_elements = M * N

        all_values = []
        all_indices = []
        batch_index = 1
        for start in range(0, total_elements, batch_size):
            print(f"processing {batch_index} batch topk")
            batch_index = batch_index+1
            end = min(start + batch_size, total_elements)

            batch = similarity_matrix.view(-1)[start:end].to(self.device)

            values, indices = torch.topk(batch, min(del_num, len(batch)), largest=False)
            values=values.to('cpu')
            indices=indices.to('cpu')
            all_values.append(values)
            all_indices.append(indices + start)

        all_values = torch.cat(all_values)
        all_indices = torch.cat(all_indices)

        final_values, final_indices = torch.topk(all_values, del_num, largest=False)

        final_indices = all_indices[final_indices]
        print("TOPK caculate success")
        return final_indices

    def modify_adj(self, adj_tensor, GB_node_list, GB_center_list, z_detached, ratio, del_ratio):


        gb_id = 1
        for cluster in GB_node_list:
            print(f"{gb_id} GB adding edges")
            gb_id += 1
            torch.cuda.empty_cache()
            gc.collect()

            cluster_tensor = torch.tensor(cluster, device=self.device)

            cluster_mask = torch.isin(adj_tensor._indices()[0], cluster_tensor) & torch.isin(adj_tensor._indices()[1],
                                                                                            cluster_tensor)

            sub_indices = adj_tensor._indices()[:, cluster_mask]
            sub_values = adj_tensor._values()[cluster_mask]
            sub_size = (len(cluster_tensor), len(cluster_tensor))

            sub_indices_remap = (
                torch.tensor([cluster_tensor.tolist().index(i.item()) for i in sub_indices[0]], dtype=torch.long,
                            device=self.device),
                torch.tensor([cluster_tensor.tolist().index(i.item()) for i in sub_indices[1]], dtype=torch.long,
                            device=self.device)
            )
            adj_submatrix = torch.sparse_coo_tensor(torch.stack(sub_indices_remap), sub_values, size=sub_size)

            z_cluster = z_detached[cluster_tensor]
            GB_sim = torch.mm(z_cluster, z_cluster.T)
            num_nodes = GB_sim.shape[0]
            add_num = min(1, int(num_nodes * ratio))

            GB_sim.fill_diagonal_(-float('inf'))

            dense_adj_submatrix = adj_submatrix.to_dense()
            GB_sim = GB_sim * (1 - dense_adj_submatrix) + dense_adj_submatrix * -float('inf')
            del dense_adj_submatrix
            torch.cuda.empty_cache()

            flat_indices = torch.topk(GB_sim.view(-1), add_num).indices
            row_indices = flat_indices // num_nodes
            col_indices = flat_indices % num_nodes

            new_indices = []
            new_values = []
            for row, col in zip(row_indices, col_indices):
                node_u = cluster_tensor[row].item()
                node_v = cluster_tensor[col].item()
                # 只添加没有边的节点对
                if not ((adj_tensor._indices()[0] == node_u) & (adj_tensor._indices()[1] == node_v)).any():
                    new_indices.append([[node_u], [node_v]])
                    new_values.append([1.0])

            if new_indices:
                new_indices = torch.tensor(new_indices, device=self.device).view(-1, 2).t()
                new_values = torch.tensor(new_values, device=self.device).view(-1)
                adj_tensor = torch.sparse_coo_tensor(
                    torch.cat((adj_tensor._indices(), new_indices), dim=1),
                    torch.cat((adj_tensor._values(), new_values), dim=0),
                    size=adj_tensor.size()
                )

            del cluster_tensor, cluster_mask, sub_indices, sub_values, sub_indices_remap, adj_submatrix, z_cluster, GB_sim, flat_indices, row_indices, col_indices, new_indices, new_values
            torch.cuda.empty_cache()
            gc.collect()

            print("========= adding edges finish ==========")

        print("Computing similarity matrix...")
        similarity_matrix = self.batched_similarity(z_detached)

        print("========= deling edges ==========")
        if del_ratio >= 1:
            all_nodes = torch.arange(adj_tensor.shape[0], device=similarity_matrix.device)
            for cluster in GB_node_list:
                cluster_nodes = torch.tensor(cluster, dtype=torch.long, device=similarity_matrix.device)
                non_cluster_nodes = all_nodes[~torch.isin(all_nodes, cluster_nodes)]
                # 簇间边置零
                adj_tensor[cluster_nodes[:, None], non_cluster_nodes] = 0
                adj_tensor[non_cluster_nodes[:, None], cluster_nodes] = 0  # 无向图保持对称
        else:

            row_indices_list = []
            col_indices_list = []
            for cluster in GB_node_list:
                cluster_tensor = torch.tensor(cluster, device=adj_tensor.device)
                row_indices_list.append(cluster_tensor.repeat_interleave(cluster_tensor.shape[0]))
                col_indices_list.append(cluster_tensor.repeat(cluster_tensor.shape[0]))

            row_indices=torch.cat(row_indices_list)
            del row_indices_list
            col_indices=torch.cat(col_indices_list)
            del col_indices_list

            mask_indices=torch.stack((row_indices,col_indices)).to('cpu')
            del row_indices,col_indices
            size=(similarity_matrix.shape[0], similarity_matrix.shape[1])
            values = torch.ones(mask_indices.shape[1], dtype=torch.bool)
            mask = torch.sparse_coo_tensor(mask_indices, values, size=size)

            inner_index=mask.coalesce().indices()
            similarity_matrix[inner_index[0],inner_index[1]] = float('inf')

            adj_tensor = adj_tensor.coalesce()
            row, col = adj_tensor.indices().to('cpu')
            saved_values = similarity_matrix[row, col]

            similarity_matrix[:] = float('inf')

            similarity_matrix[row, col] = saved_values

            total_edges = adj_tensor._nnz() // 2
            del_num = int(total_edges * del_ratio)

            flat_indices = self.batched_topk(similarity_matrix, del_num,batch_size = 100000000)
            row_indices = flat_indices // adj_tensor.shape[0]
            col_indices = flat_indices % adj_tensor.shape[0]
            del similarity_matrix

            before_nnz = adj_tensor._nnz()
            adj_tensor_dense=adj_tensor.to('cpu').to_dense()

            adj_tensor_dense[row_indices,col_indices] = 0
            adj_tensor = adj_tensor_dense.to_sparse_coo().to(self.device)
            del_edge_count =  before_nnz - adj_tensor._nnz()
            del adj_tensor_dense
            print("========= del edges finish ==========")

            adj_tensor = adj_tensor.coalesce()
            return csr_matrix((adj_tensor.values().cpu().numpy(),
                               adj_tensor.indices().cpu().numpy()),
                              shape=adj_tensor.shape)

    def batched_similarity(self, z, batch_size=1024 * 2):
        n_nodes = z.shape[0]
        CHUNK = batch_size
        CTS = (n_nodes + CHUNK - 1) // CHUNK

        rows = []
        cols = []
        values = []
        similarity_matrix=torch.empty(n_nodes,0,device='cpu')

        print("========== calculate sim matrix ==============")
        for j in range(CTS):
            gc.collect()
            print(f"calculating {j} sim matrix。。。。。。")
            a = j * CHUNK
            b = (j + 1) * CHUNK
            b = min(b, n_nodes)

            cts = torch.matmul(z, z[a:b].T).cpu()
            similarity_matrix=torch.cat((similarity_matrix,cts),dim=1)

        print("==========sim matrix finish==============")
        return similarity_matrix
    def get_embedding(self):

        with torch.no_grad():
            mu = self.best_model.encoder(self.sm_fea_s)
        return mu.detach()



def data_split(full_list, n_sample):

    offset = n_sample
    random.shuffle(full_list)
    len_all = len(full_list)
    index_now = 0
    split_list = []
    while index_now < len_all:
        if index_now + offset > len_all:
            split_list.append(full_list[index_now:len_all])
        else:
            split_list.append(full_list[index_now:index_now + offset])
        index_now += offset
    return split_list

