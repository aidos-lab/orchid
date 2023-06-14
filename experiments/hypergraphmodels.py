import gzip
import json
import math
import os
from itertools import chain

import numpy as np
from scipy.sparse import csr_matrix


class HypergraphConfigurationModel:
    def __init__(self, node_degrees, edge_cardinalities, seed, identifier=None):
        assert sum(node_degrees) == sum(
            edge_cardinalities
        ), f"Sum of node degrees does not equal sum of edge cardinalities! {sum(node_degrees)} != {sum(edge_cardinalities)}"
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.input_node_degrees = node_degrees
        self.input_edge_cardinalities = edge_cardinalities
        self.M_csr = self._generate_edges()
        self.M_csc = self.M_csr.tocsc()
        self.n = len(node_degrees)
        self.m = len(edge_cardinalities)
        self.node_degree = [int(self.M_csr[i, :].nnz) for i in range(self.n)]
        self.edge_cardinality = [int(self.M_csc[:, j].nnz) for j in range(self.m)]
        self.node_neighborhood_size = [
            len(set(self.M_csc[:, self.M_csr[i, :].nonzero()[-1]].nonzero()[0]))
            for i in range(self.M_csr.shape[0])
        ]
        self.edge_neighborhood_size = [
            len(set(self.M_csr[self.M_csc[:, j].nonzero()[0]].nonzero()[-1]))
            for j in range(self.M_csc.shape[-1])
        ]
        self.c = sum(self.node_degree)
        self.identifier = identifier

    def _generate_edges(self):
        row_ind = list(
            chain.from_iterable([n] * d for n, d in enumerate(self.input_node_degrees))
        )
        col_ind = list(
            chain.from_iterable(
                [n] * d for n, d in enumerate(self.input_edge_cardinalities)
            )
        )
        data = [1] * len(col_ind)
        self.random_state.shuffle(row_ind)
        self.random_state.shuffle(col_ind)
        # csr_matrix ignores duplicate entries
        return csr_matrix((data, (row_ind, col_ind)))

    def _get_filename(self):
        return f"HCM_n-{self.n}_m-{self.m}_c-{self.c}_seed-{self.seed}"

    def _generate_ihg_tsv_string(self):
        c2r = [
            list(map(lambda x: int(x + 1), self.M_csc[:, c].nonzero()[0]))
            for c in range(self.M_csc.shape[-1])
        ]
        if self.identifier is None:
            return "\n".join(["\t".join(map(str, e)) for e in c2r])
        else:
            return "\n".join(
                [f"{self.identifier}\t" + "\t".join(map(str, e)) for e in c2r]
            )

    def write_ihg_tsv_gz(self, savepath):
        c2r_string = self._generate_ihg_tsv_string()
        os.makedirs(savepath, exist_ok=True)
        with gzip.open(
            f"{savepath}/{self._get_filename()}.ihg.tsv.gz", "wt", encoding="UTF-8"
        ) as zipfile:
            zipfile.write(c2r_string)

    def _generate_ihg_features(self):
        features = {
            "node_degree": self.node_degree,
            "edge_cardinality": self.edge_cardinality,
            "node_neighborhood_size": self.node_neighborhood_size,
            "edge_neighborhood_size": self.edge_neighborhood_size,
            "config": dict(
                seed=self.seed,
                original_node_degrees=self.input_node_degrees,
                original_edge_cardinalities=self.input_edge_cardinalities,
            ),
            "filename": self._get_filename(),
        }
        if self.identifier is not None:
            features["identifier"] = self.identifier
        return features

    def write_ihg_features_gz(self, savepath):
        with gzip.open(
            f"{savepath}/{self._get_filename()}.ihg.json.gz", "wt", encoding="UTF-8"
        ) as zipfile:
            json.dump(
                self._generate_ihg_features(),
                zipfile,
            )

    def __repr__(self):
        return f"<HypergraphConfigurationModel with {self.n} nodes and {self.m} edges>"


class HnmpModel:
    def __init__(self, n, m, p, seed, identifier=None):
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.n = n
        self.m = m
        self.p = p
        self.c = int(round(p * n * m))
        self.M_csr = self._generate_edges()
        self.M_csc = self.M_csr.tocsc()
        self.node_degree = [
            int(self.M_csr[i, :].nnz) for i in range(self.M_csr.shape[0])
        ]
        self.edge_cardinality = [
            int(self.M_csc[:, j].nnz) for j in range(self.M_csr.shape[-1])
        ]
        self.node_neighborhood_size = [
            len(set(self.M_csc[:, self.M_csr[i, :].nonzero()[-1]].nonzero()[0]))
            for i in range(self.M_csr.shape[0])
        ]
        self.edge_neighborhood_size = [
            len(set(self.M_csr[self.M_csc[:, j].nonzero()[0]].nonzero()[-1]))
            for j in range(self.M_csc.shape[-1])
        ]
        self.identifier = identifier

    def _generate_edges(self):
        row_ind = list(
            map(
                int, self.random_state.choice(list(range(self.n)), self.c, replace=True)
            )
        )
        col_ind = list(
            map(
                int, self.random_state.choice(list(range(self.m)), self.c, replace=True)
            )
        )
        data = [1] * len(col_ind)
        # csr_matrix ignores duplicate entries
        return csr_matrix((data, (row_ind, col_ind)))

    def _generate_ihg_tsv_string(self):
        c2r = [
            list(map(lambda x: int(x + 1), self.M_csc[:, c].nonzero()[0]))
            for c in range(self.M_csc.shape[-1])
        ]
        if self.identifier is None:
            return "\n".join(["\t".join(map(str, e)) for e in c2r])
        else:
            return "\n".join(
                [f"{self.identifier}\t" + "\t".join(map(str, e)) for e in c2r]
            )

    def write_ihg_tsv_gz(self, savepath):
        c2r_string = self._generate_ihg_tsv_string()
        os.makedirs(savepath, exist_ok=True)
        with gzip.open(
            f"{savepath}/{self._get_filename()}.ihg.tsv.gz", "wt", encoding="UTF-8"
        ) as zipfile:
            zipfile.write(c2r_string)

    def _generate_ihg_features(self):
        features = {
            "node_degree": self.node_degree,
            "edge_cardinality": self.edge_cardinality,
            "node_neighborhood_size": self.node_neighborhood_size,
            "edge_neighborhood_size": self.edge_neighborhood_size,
            "config": dict(
                seed=self.seed,
                n=self.n,
                m=self.m,
                p=self.p,
            ),
            "filename": self._get_filename(),
        }
        if self.identifier is not None:
            features["identifier"] = self.identifier
        return features

    def write_ihg_features_gz(self, savepath):
        with gzip.open(
            f"{savepath}/{self._get_filename()}.ihg.json.gz", "wt", encoding="UTF-8"
        ) as zipfile:
            json.dump(
                self._generate_ihg_features(),
                zipfile,
            )

    def _get_filename(self):
        return f"Hnmp_n-{self.n}_m-{self.m}_c-{self.c}_seed-{self.seed}"

    def __repr__(self):
        return f"<H(n,m,p) Model with {self.n} nodes and {self.m} edges and p={round(self.p,8)}>"


class HSBModel:
    def __init__(
        self,
        node_community_sizes,
        edge_community_sizes,
        affinity_matrix,
        seed,
        with_hash=False,
        identifier=None,
    ):
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.node_community_sizes = node_community_sizes
        self.edge_community_sizes = edge_community_sizes
        self.affinity_matrix = affinity_matrix
        self.n_node_communities = len(self.node_community_sizes)
        self.n_edge_communities = len(self.edge_community_sizes)
        self.node_communities = list(
            chain.from_iterable(
                [n] * d for n, d in enumerate(self.node_community_sizes)
            ),
        )
        self.edge_communities = list(
            chain.from_iterable(
                [n] * d for n, d in enumerate(self.edge_community_sizes)
            ),
        )
        self.M_csr = self._generate_edges()
        self.M_csc = self.M_csr.tocsc()
        self.n = int(self.M_csr.shape[0])
        self.m = int(self.M_csr.shape[-1])
        self.c = int(self.M_csr.nnz)
        self.with_hash = with_hash
        self.node_degree = [int(self.M_csr[i, :].nnz) for i in range(self.n)]
        self.edge_cardinality = [int(self.M_csc[:, j].nnz) for j in range(self.m)]
        self.node_neighborhood_size = [
            len(set(self.M_csc[:, self.M_csr[i, :].nonzero()[-1]].nonzero()[0]))
            for i in range(self.M_csr.shape[0])
        ]
        self.edge_neighborhood_size = [
            len(set(self.M_csr[self.M_csc[:, j].nonzero()[0]].nonzero()[-1]))
            for j in range(self.M_csc.shape[-1])
        ]
        self.identifier = identifier

    def _generate_edges(self):
        row_ind = list()
        col_ind = list()
        for node_idx, v in enumerate(self.node_communities):
            affinities = self.affinity_matrix[v, :]
            for comm_idx, (community_size, affinity) in enumerate(
                zip(self.edge_community_sizes, affinities)
            ):
                n_edges_to_sample = int(
                    self.random_state.choice([math.ceil, math.floor])(
                        affinity * community_size
                    )
                )
                if n_edges_to_sample > 0:
                    edges_to_choose_from = [
                        idx
                        for idx, e in enumerate(self.edge_communities)
                        if e == comm_idx
                    ]
                    edges_sampled = self.random_state.choice(
                        edges_to_choose_from, size=n_edges_to_sample, replace=False
                    )
                    row_ind.extend([node_idx] * len(edges_sampled))
                    col_ind.extend(edges_sampled)
        data = [1] * len(col_ind)
        # csr_matrix ignores duplicate entries
        return csr_matrix((data, (row_ind, col_ind)))

    def _generate_ihg_tsv_string(self):
        c2r = [
            list(map(lambda x: int(x + 1), self.M_csc[:, c].nonzero()[0]))
            for c in range(self.M_csc.shape[-1])
        ]
        if self.identifier is None:
            return "\n".join(["\t".join(map(str, e)) for e in c2r])
        else:
            return "\n".join(
                [f"{self.identifier}\t" + "\t".join(map(str, e)) for e in c2r]
            )

    def write_ihg_tsv_gz(self, savepath):
        c2r_string = self._generate_ihg_tsv_string()
        os.makedirs(savepath, exist_ok=True)
        with gzip.open(
            f"{savepath}/{self._get_filename()}.ihg.tsv.gz", "wt", encoding="UTF-8"
        ) as zipfile:
            zipfile.write(c2r_string)

    def _generate_ihg_features(self):
        features = {
            "node_degree": self.node_degree,
            "edge_cardinality": self.edge_cardinality,
            "node_neighborhood_size": self.node_neighborhood_size,
            "edge_neighborhood_size": self.edge_neighborhood_size,
            "node_communities": list(map(lambda x: x + 1, self.node_communities)),
            "edge_communities": list(map(lambda x: x + 1, self.edge_communities)),
            "config": dict(
                seed=self.seed,
                node_community_sizes=self.node_community_sizes,
                edge_community_sizes=self.edge_community_sizes,
                affinity_matrix=self.affinity_matrix.tolist(),
            ),
            "filename": self._get_filename(),
        }
        if self.identifier is not None:
            features["identifier"] = self.identifier
        return features

    def write_ihg_features_gz(self, savepath):
        with gzip.open(
            f"{savepath}/{self._get_filename()}.ihg.json.gz", "wt", encoding="UTF-8"
        ) as zipfile:
            json.dump(
                self._generate_ihg_features(),
                zipfile,
            )

    def _get_filename(self):
        return f"HSBM_n-{self.n}_m-{self.m}_c-{self.c}_nnc-{self.n_node_communities}_nmc-{self.n_edge_communities}_seed-{self.seed}{'' if not self.with_hash else '_h-' + str(hash(str(self.seed) + str(self.affinity_matrix.tolist())))}"

    def __repr__(self):
        return f"<Hypergraph Stochastic Block Model with {self.n} nodes and {self.m} edges, {self.n_node_communities} node communities and {self.n_edge_communities} edge communities>"
