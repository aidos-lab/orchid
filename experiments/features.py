import argparse
import gzip
import json
from collections import OrderedDict

from scipy.sparse import csr_matrix


def compute_edge_cardinalities(M_csc):
    edge_cardinalities = [int(M_csc[:, j].nnz) for j in range(M_csc.shape[-1])]
    return edge_cardinalities


def compute_edge_neighborhood_sizes(M_csr, M_csc):
    edge_neighborhood_sizes = [
        len(set(M_csr[M_csc[:, j].nonzero()[0]].nonzero()[-1]))
        for j in range(M_csc.shape[-1])
    ]
    return edge_neighborhood_sizes


def compute_node_degrees(M_csr):
    node_degrees = [int(M_csr[i, :].nnz) for i in range(M_csr.shape[0])]
    return node_degrees


def compute_node_neighborhood_sizes(M_csr, M_csc):
    node_neighborhood_sizes = [
        len(set(M_csc[:, M_csr[i, :].nonzero()[-1]].nonzero()[0]))
        for i in range(M_csr.shape[0])
    ]
    return node_neighborhood_sizes


def compute_ihg_features(hypergraph_edges, input_file):
    row_ind = [item for sublist in hypergraph_edges for item in sublist]
    col_ind = [
        item
        for sublist in [
            [idx] * len(sublist) for idx, sublist in enumerate(hypergraph_edges)
        ]
        for item in sublist
    ]
    data = [1] * len(col_ind)
    M_csr = csr_matrix((data, (row_ind, col_ind)))
    M_csc = M_csr.tocsc()
    result = {
        "edge_cardinality": compute_edge_cardinalities(M_csc),
        "edge_neighborhood_size": compute_edge_neighborhood_sizes(M_csr, M_csc),
        "node_degree": compute_node_degrees(M_csr),
        "node_neighborhood_size": compute_node_neighborhood_sizes(M_csr, M_csc),
        "input": input_file,
    }
    return result


def compute_chg_features(hypergraph_collection_edges, input_file):
    result = [
        compute_ihg_features(hypergraph_edges, input_file)
        for hypergraph_edges in hypergraph_collection_edges
    ]
    return result


def edge_to_nodes_zero_indexed(edge):
    return [(int(node) - 1) for node in edge.strip().split("\t")]


def edge_strings_to_graph_collection(edge_strings):
    graphs = OrderedDict()
    for e in edge_strings:
        edge = edge_to_nodes_zero_indexed(e)
        if edge[0] not in graphs:
            graphs[edge[0]] = [edge[1:]]
        else:
            graphs[edge[0]].append(edge[1:])
    edges = list(graphs.values())
    return edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input {ihg,chg} file"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output destination"
    )
    args = parser.parse_args()
    i = args.input
    o = args.output

    with (gzip.open(i, "rt") if ".gz" in i else open(i, "r")) as f:
        edge_strings = f.read().strip().split("\n")

    if "ihg.tsv" in i:
        edges = [edge_to_nodes_zero_indexed(edge) for edge in edge_strings]
        result = compute_ihg_features(edges, i)
    else:
        edges = edge_strings_to_graph_collection(edge_strings)
        result = compute_chg_features(edges, i)

    print(
        json.dumps(result),
        file=(gzip.open(o, "wt") if ".gz" in o else open(o, "wt")),
    )
