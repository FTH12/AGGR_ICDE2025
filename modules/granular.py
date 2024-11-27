import math
import torch
import networkx as nx
from scipy.stats import pearsonr


class Granular():
    def __init__(self,quity='homo',sim='dot'):
        self.labels = None
        self.adj_tensor = None
        self.center_cluster = None
        self.z_detached = None
        self.gb_q = []
        self.predsoft = None
        self.methods = dict()
        self.methods['quity'] = quity
        self.methods['sim'] = sim
        pass
    def process_graph(self,adj):
        graph = nx.from_scipy_sparse_array(adj)
        return graph


    def get_sub_adj_z(self, sub_g, cluster):
        node_indices = torch.tensor(cluster)

        sub_adj = torch.tensor(nx.to_scipy_sparse_array(sub_g, format='csr').toarray())
        sub_z = self.z_detached[node_indices]
        return sub_adj, sub_z

    def quity(self, adj_s, z_detach):
        num_edges = torch.sum(adj_s) // 2

        x = torch.matmul(torch.t(z_detach).double(), adj_s.double().to(z_detach.device))
        x = torch.matmul(x, z_detach.double())
        x = torch.trace(x)  # Tr(BXX^T)

        degree_s = adj_s.sum(dim=1)
        y = torch.matmul(torch.t(z_detach).double(), degree_s.double().to(z_detach.device))
        y = (y ** 2).sum() / (2 * num_edges)
        quality = ((x - y) / (2 * num_edges))
        return quality

    def quity_degree(self, adj_s):
        total_edges = adj_s.sum() // 2

        num_nodes = adj_s.shape[0]

        edge_to_node_ratio = total_edges / num_nodes

        return edge_to_node_ratio

    def quity_homo(self, adj_s, z_detach):
        sim_matrix = torch.mm(z_detach, z_detach.T)
        sim_matrix.fill_diagonal_(0)
        weighted_sim_matrix = adj_s.to(sim_matrix.device) * sim_matrix

        total_similarity = weighted_sim_matrix.sum()
        num_edges = adj_s.sum() / 2
        avg_homogeneity = total_similarity / num_edges
        return avg_homogeneity
    def quity_edges(self, adj_s):
        sub_edges = adj_s.sum() // 2
        degree_sum = adj_s.sum(dim=1).sum().item()
        m_exp = (degree_sum ** 2) / (4 * self.total_edges)
        Q_s = (sub_edges / self.total_edges) - (m_exp / self.total_edges)
        return Q_s

    def get_quity(self,adj_s,z_detach=None):
        if adj_s.shape[0] == 1:
            return torch.tensor(0)
        if self.methods['quity'] == 'detach':
            return self.quity(adj_s, z_detach)
        elif self.methods['quity'] == 'homo':
            return self.quity_homo(adj_s, z_detach)
        elif self.methods['quity'] == 'edges':
            return self.quity_edges(adj_s)
        elif self.methods['quity'] == 'deg':
            return self.quity_degree(adj_s)

    def get_sim(self, node1, node2):
        emb_node1 = self.z_detached[node1]
        emb_node2 = self.z_detached[node2]
        if self.methods['sim']=='dot':
            return self.dot_sim(emb_node1,emb_node2)
        elif self.methods['sim']=='cos':
            return self.cos_sim(emb_node1,emb_node2)
        elif self.methods['sim'] == 'per':
            return  self.per_sim(emb_node1,emb_node2)
        else:
            return self.dot_sim(emb_node1, emb_node2)
    def dot_sim(self, emb1,emb2):

        return torch.dot(emb1,emb2)
    def cos_sim(self, emb1,emb2):
        return torch.nn.functional.cosine_similarity(emb1, emb2, dim=0)

    def per_sim(self, emb1, emb2):
        emb1_np = emb1.cpu().numpy()
        emb2_np = emb2.cpu().numpy()
        similarity, _ = pearsonr(emb1_np, emb2_np)
        return similarity

    def init_GB(self, graph:nx.Graph):
        degree_dict = dict(graph.degree())
        sorted_nodes = sorted(degree_dict, key=degree_dict.get)
        center_nodes = sorted_nodes[-self.init_num:]
        points = set(sorted_nodes[0:-self.init_num])
        clusters = []
        neighbors_list = []
        clusters_len = []
        for node in center_nodes:
            clusters.append([node])
            neighbors_list.append(list(graph.neighbors(node)))
            clusters_len.append(1)
        located_point = set()

        point_to_cluster = dict()
        sim_dic = dict()
        while len(points) > 0:
            for i in range(self.init_num):
                for neighbor in neighbors_list[i]:
                    if neighbor in points:
                        if point_to_cluster.get(neighbor) is None:
                            point_to_cluster[neighbor] = i
                            sim_dic[neighbor] = self.get_sim(neighbor, center_nodes[i])
                        else:
                            old_sim = sim_dic[neighbor]
                            new_sim = self.get_sim(neighbor, center_nodes[i])
                            if old_sim < new_sim:
                                point_to_cluster[neighbor] = i
                                sim_dic[neighbor] = new_sim
            new_neighbors = []
            for i in range(self.init_num):
                tset = set()
                new_neighbors.append(tset)
            for point in point_to_cluster:
                idx = point_to_cluster[point]
                clusters[idx].append(point)
                clusters_len[idx] += 1
                located_point.add(point)
                new_neighbors[idx].update(list(graph.neighbors(point)))
            neighbors_list.clear()
            neighbors_list = new_neighbors
            point_to_cluster.clear()
            points -= located_point
            located_point.clear()
        cluster_Q = []

        init_GB_list = [graph.subgraph(cluster) for cluster in clusters]
        for idx,cluster in enumerate(clusters):
            sub_adj, sub_z = self.get_sub_adj_z(init_GB_list[idx], cluster)
            sub_q = self.get_quity(sub_adj, sub_z)
            cluster_Q.append(sub_q)
        return init_GB_list, clusters, cluster_Q, center_nodes


    def split_bfs(self,graph, split_GB_list, split_graph_list,split_center_list, center_f, quality_f):
        node_num = graph.number_of_nodes()
        if node_num <= 3:
            split_GB_list.append(list(graph.nodes()))
            split_graph_list.append(graph)
            split_center_list.append(center_f)
            self.gb_q.append(quality_f)
            return
        degree_dict = dict(graph.degree())
        sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
        center_nodes = [sorted_nodes[1], sorted_nodes[2]]

        points = set(sorted_nodes)
        points.remove(center_nodes[0])
        points.remove(center_nodes[1])

        clusters = [[center_nodes[0]], [center_nodes[1]]]
        neighbors_list = [set(graph.neighbors(center_nodes[0])), set(graph.neighbors(center_nodes[1]))]
        common_neighbors = neighbors_list[0] & neighbors_list[1]

        while len(points) > 0:
            new_neighbors = [set() , set()]
            for neighbor in common_neighbors:
                if neighbor in points:
                    points.remove(neighbor)
                    sim_1 = self.get_sim(neighbor,center_nodes[0])
                    sim_2 = self.get_sim(neighbor,center_nodes[1])
                    if sim_1 > sim_2:
                        to_idx = 0
                    else:
                        to_idx = 1
                    clusters[to_idx].append(neighbor)
                    new_neighbors[to_idx].update(graph.neighbors(neighbor))
            for i in range(2):
                for neighbor in neighbors_list[i]:
                    if neighbor in points:
                        points.remove(neighbor)
                        clusters[i].append(neighbor)
                        new_neighbors[i].update(graph.neighbors(neighbor))
            neighbors_list.clear()
            neighbors_list = new_neighbors
            common_neighbors = neighbors_list[0] & neighbors_list[1]

        if len(clusters[0]) < 2 or len(clusters[1]) < 2:
            split_GB_list.append(list(graph.nodes()))
            split_graph_list.append(graph)
            self.gb_q.append(quality_f)
            split_center_list.append(center_f)
            return
        subgraph_a = graph.subgraph(clusters[0])
        subgraph_b = graph.subgraph(clusters[1])
        if (not nx.is_connected(subgraph_a)) or (not nx.is_connected(subgraph_b)):
            split_GB_list.append(list(graph.nodes()))
            split_graph_list.append(graph)
            self.gb_q.append(quality_f)
            split_center_list.append(center_f)
            return
        adj_a,z_a = self.get_sub_adj_z(subgraph_a, clusters[0])
        adj_b,z_b = self.get_sub_adj_z(subgraph_b, clusters[1])
        quality_a = self.get_quity(adj_a, z_a)
        quality_b = self.get_quity(adj_b, z_b)

        if quality_f > (quality_a + quality_b)/2.5:
            split_GB_list.append(list(graph.nodes()))
            split_graph_list.append(graph)
            self.gb_q.append(quality_f)
            split_center_list.append(center_f)
            return
        else:
            self.split_bfs(subgraph_a, split_GB_list, split_graph_list,split_center_list,center_nodes[0], quality_a)
            self.split_bfs(subgraph_b, split_GB_list, split_graph_list,split_center_list,center_nodes[1], quality_b)

    @staticmethod
    def get_node_subgraph_edges(graph, node, subgraph):
        edge_count = 0
        for point in subgraph:
            if graph.has_edge(node, point):
                edge_count += 1

        return edge_count
    def get_GB_graph(self,graph):
        init_GB_num = math.isqrt(len(graph))
        self.init_num = int(init_GB_num)
        init_GB_list, clusters, cluster_Q, init_center = self.init_GB(graph)

        GB_list = []

        GB_graph_list = []

        GB_center_list = []
        for i, init_GB in enumerate(init_GB_list):
            split_GB_list = []
            split_graph_list = []
            split_center_list = []
            self.split_bfs(init_GB, split_GB_list, split_graph_list,split_center_list, init_center[i], cluster_Q[i])
            GB_list.extend(split_GB_list)
            GB_graph_list.extend(split_graph_list)
            GB_center_list.extend(split_center_list)
        return GB_list, GB_graph_list,GB_center_list
    def generate_GB(self, graph):
        GB_node_list = []
        GB_graph_list = []
        GB_center_list = []
        if nx.is_connected(graph):
            GB_node_list, GB_graph_list,GB_center_list = self.get_GB_graph(graph)
        else:
            connected_components = nx.connected_components(graph)
            for component in connected_components:
                subgraph = graph.subgraph(component)
                if len(subgraph) <= 3:
                    degree_dict = dict(subgraph.degree())
                    max_node = max(degree_dict, key=degree_dict.get)
                    GB_graph_list.append(subgraph)
                    GB_node_list.append(list(subgraph.nodes()))
                    GB_center_list.append(max_node)
                else:
                    node_list, graph_list,center_list = self.get_GB_graph(subgraph)
                    GB_node_list.extend(node_list)
                    GB_graph_list.extend(graph_list)
                    GB_center_list.extend(center_list)
        return GB_node_list, GB_graph_list, GB_center_list
    def forward(self, adj_csr):
        graph = self.process_graph(adj_csr)
        self.total_edges = graph.number_of_nodes()
        GB_node_list, GB_graph_list, GB_center_list = self.generate_GB(graph)

        return GB_node_list, GB_graph_list,GB_center_list
    def forward_batch(self, adj_csr):
        graph = self.process_graph(adj_csr)
        GB_node_list, GB_graph_list = self.generate_GB(graph)
        return GB_node_list, GB_graph_list

