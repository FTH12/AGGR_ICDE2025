"""
Adaptive Granular Graph Rewiring via Granular-ball for graph clustering
"""
import copy
import random
import time
from typing import Callable, List
from utils.utils import check_modelfile_exists,load_model,save_model
import dgl
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from torch import nn

from modules import InnerProductDecoder, LinTrans, preprocess_graph, SampleDecoder
from modules.granular import Granular
from utils import eliminate_zeros, set_seed

class AGGR(nn.Module):

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
    def get_fd_loss(z: torch.Tensor, norm: int = 1) -> torch.Tensor:
        norm_ff = z / (z**2).sum(0, keepdim=True).sqrt()
        coef_mat = torch.mm(norm_ff.t(), norm_ff)
        coef_mat.div_(2.0)
        a = torch.arange(coef_mat.size(0), device=coef_mat.device)
        L_fd = norm * F.cross_entropy(coef_mat, a)
        return L_fd

    @staticmethod
    def target_distribution(q):
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def get_cluster_center(self, z=None):
        if z is None:
            z = self.get_embedding()
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        _ = kmeans.fit_predict(z.data.cpu().numpy())
        self.cluster_layer.data = torch.Tensor(kmeans.cluster_centers_).to(self.device)

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / 1)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


    def _pretrain(self):
        best_loss = 1e9

        for epoch in range(self.n_pretrain_epochs):
            self.train()
            self.optimizer.zero_grad()
            t = time.time()

            z = self.encoder(self.sm_fea_s)
            preds = self.inner_product_decoder(z).view(-1)
            loss = self.bce_loss(preds, self.lbls, norm=self.norm_weights, pos_weight=self.pos_weight)
            (loss + self.get_fd_loss(z, self.regularization)).backward()
            self.optimizer.step()
            cur_loss = loss.item()
            print(f"Cluster Epoch: {epoch}, embeds_loss={cur_loss} time={time.time() - t:.5f}")
            if cur_loss < best_loss:
                best_loss = cur_loss
                del self.best_model
                self.best_model = copy.deepcopy(self.to(self.device))

    def _train(self):
        best_loss = 1e9

        self.get_cluster_center()
        for epoch in range(self.n_epochs):
            self.train()

            if epoch % self.udp == 0 or epoch == self.n_epochs - 1:
                with torch.no_grad():
                    z_detached = self.get_embedding()
                    Q = self.get_Q(z_detached)
                    q = Q.detach().data.cpu().numpy().argmax(1)
            self.optimizer.zero_grad()
            t = time.time()

            z = self.encoder(self.sm_fea_s)
            preds = self.inner_product_decoder(z).view(-1)
            loss = self.bce_loss(preds, self.lbls, self.norm_weights, self.pos_weight)

            q = self.get_Q(z)
            p = self.target_distribution(Q.detach())
            kl_loss = F.kl_div(q.log(), p, reduction="batchmean")
            aux_loss = self.aux_objective(z)
            (
               aux_loss +
                loss + kl_loss + self.regularization * self.bce_loss(
                self.inner_product_decoder(q).view(-1), preds)).backward()

            self.optimizer.step()
            cur_loss = loss.item()

            print(f"Cluster Epoch: {epoch}, embeds_loss={cur_loss:.5f},"
                  f"kl_loss={kl_loss.item()},"
                  f"time={time.time() - t:.5f},"
                  f"aux_loss={aux_loss.item()},"
                  )
            if cur_loss < best_loss:
                best_loss = cur_loss
                del self.best_model
                self.best_model = copy.deepcopy(self.to(self.device))

    def update_adj(
        self,
        adj,
        ratio=0.2,
        del_ratio=0.005,
    ):
        adj_tensor = torch.IntTensor(adj.toarray())
        self.to(self.device)
        z_detached = self.get_embedding()
        self.gb_layer.labels = self.labels
        self.gb_layer.adj_tensor = adj_tensor.to(self.device)
        self.gb_layer.z_detached = z_detached
        GB_node_list, GB_graph_list,GB_center_list = self.gb_layer.forward(adj)

        adj_new = self.modify_adj(adj_tensor.to(self.device), GB_node_list,GB_center_list, z_detached, ratio, del_ratio)
        return adj_new

    def modify_adj(self, adj_tensor, GB_node_list, GB_center_list, z_detached, ratio, del_ratio):

        if ratio == 1:
            for cluster in GB_node_list:
                cluster_nodes = torch.tensor(cluster, dtype=torch.long)
                adj_tensor[cluster_nodes[:, None], cluster_nodes] = 1
        else:
            for cluster in GB_node_list:
                cluster_tensor = torch.tensor(cluster)
                GB_sim = torch.mm(z_detached[cluster_tensor], z_detached[cluster_tensor].T)
                num_nodes = GB_sim.shape[0]
                add_num = min(1, int(num_nodes * ratio))
                GB_sim.fill_diagonal_(-float('inf'))

                adj_submatrix = adj_tensor[cluster_tensor][:, cluster_tensor].to_dense()
                GB_sim = GB_sim * (1 - adj_submatrix) + adj_submatrix * -float('inf')

                flat_indices = torch.topk(GB_sim.view(-1), add_num).indices
                row_indices = flat_indices // num_nodes
                col_indices = flat_indices % num_nodes

                for row, col in zip(row_indices, col_indices):
                    node_u = cluster_tensor[row].item()
                    node_v = cluster_tensor[col].item()
                    if adj_tensor[node_u, node_v] == 0:
                        adj_tensor[node_u, node_v] = 1
                        adj_tensor[node_v, node_u] = 1

        if del_ratio >= 1:
            all_nodes = torch.arange(adj_tensor.shape[0], device=adj_tensor.device)
            for cluster in GB_node_list:
                cluster_nodes = torch.tensor(cluster, dtype=torch.long, device=adj_tensor.device)
                non_cluster_nodes = all_nodes[~torch.isin(all_nodes, cluster_nodes)]
                adj_tensor[cluster_nodes[:, None], non_cluster_nodes] = 0
                adj_tensor[non_cluster_nodes[:, None], cluster_nodes] = 0
        else:
            similarity_matrix = torch.mm(z_detached, z_detached.T)

            mask = torch.ones_like(similarity_matrix, dtype=torch.bool)
            for cluster in GB_node_list:
                cluster_tensor = torch.tensor(cluster)
                mask[cluster_tensor[:, None], cluster_tensor] = False

            similarity_matrix[~mask] = float('inf')

            similarity_matrix[adj_tensor == 0] = float('inf')

            total_edges = adj_tensor.sum() // 2
            del_num = int(total_edges * del_ratio)
            flat_indices = torch.topk(similarity_matrix.view(-1), del_num, largest=False).indices
            row_indices = flat_indices // adj_tensor.shape[0]
            col_indices = flat_indices % adj_tensor.shape[0]

            for row, col in zip(row_indices, col_indices):
                if adj_tensor[row, col] == 1:
                    adj_tensor[row, col] = 0
                    adj_tensor[col, row] = 0

        isolated_nodes = torch.where(adj_tensor.sum(dim=1) == 0)[0]
        if len(isolated_nodes) > 0:
            for node_zero in isolated_nodes:
                with torch.no_grad():
                    non_isolated_sim = similarity_matrix[node_zero].clone()
                    non_isolated_sim[isolated_nodes] = -float('inf')
                    max_sim_node = non_isolated_sim.argmax().item()
                    adj_tensor[node_zero, max_sim_node] = 1
                    adj_tensor[max_sim_node, node_zero] = 1
        adj_tensor.fill_diagonal_(0)
        adj_matrix = csr_matrix(adj_tensor.cpu().numpy())

        return adj_matrix
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
        self.lbls = torch.FloatTensor(adj_csr.todense()).view(-1).to(self.device)

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
    ) -> None:
        self.device = device
        self.features = graph.ndata["feat"]
        adj = self.adj_orig = graph.adj_external(scipy_fmt="csr")
        self.n_nodes = self.features.shape[0]
        self.labels = labels
        self.gb_layer = Granular(quity=quilty, sim=simm)
        self.adj_sum_raw = adj_sum_raw
        adj = eliminate_zeros(adj)
        self.to(self.device)
        self.update_features(adj,True)
        adj = self.adj_label

        if load and check_modelfile_exists(self.warmup_filename):
            self, self.optimizer, _, _ = load_model(
                self.warmup_filename,
                self,
                self.optimizer,
                self.device,
            )
            self.to(self.device)
            print(f"model loaded from {self.warmup_filename} to {self.device}")
        else:
            self._pretrain()
            if dump:
                save_model(
                    self.warmup_filename,
                    self,
                    self.optimizer,
                    None,
                    None,
                )
                print(f"dump to {self.warmup_filename}")
        self.on_labels = F.one_hot(self.labels, num_classes=self.n_clusters).to(self.device)
        if gr_epochs != 0:
            self._train()
            with torch.no_grad():
                z_detached = self.get_embedding()
                Q = self.get_Q(z_detached)
                q = Q.argmax(dim=1)
                self.on_labels = F.one_hot(q, num_classes=Q.shape[1])
        adj_pre = copy.deepcopy(adj)

        for gr_ep in range(1, gr_epochs + 1):
            print(f"==============GR epoch:{gr_ep} ===========")

            adj_new = eliminate_zeros(
                self.update_adj(
                    adj_pre,
                    ratio=node_ratio,
                    del_ratio=del_edge_ratio,
                ))
            self.update_features(adj=adj_new)

            self._train()
            adj_pre = copy.deepcopy(self.adj_label)
            with torch.no_grad():
                z_detached = self.get_embedding()
                Q = self.get_Q(z_detached)
                q = Q.argmax(dim=1)
                self.on_labels = F.one_hot(q, num_classes=Q.shape[1])

            if self.reset:
                self.reset_weights()
                self.to(self.device)

    def get_embedding(self, best=True):
        with torch.no_grad():
            mu = (self.best_model.encoder(self.sm_fea_s) if best else self.encoder(self.sm_fea_s))
        return mu.detach()
