import argparse
import gzip
import json
from glob import glob

import numpy as np
from common import *
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import scale


def kpca(histogram, values, k, feature_name, gammas=[1]):
    results = []
    for gamma in gammas:
        print("kPCA EWK")
        pca = KernelPCA(n_components=k, kernel="precomputed", tol=1e-5, max_iter=2000)
        G = pairwise_distances(
            values, metric=lambda a, b: wasserstein_kernel(a, b, gamma), n_jobs=-1
        )
        embedding = pca.fit_transform(G)
        results.append(
            {
                "method": "KernelPCA",
                "kernel": "wasserstein_kernel",
                "kernel_hyperparameter": gamma,
                "ncomponents": k,
                "feature_name": feature_name,
                "embedding": [e.tolist() for e in embedding],
            }
        )
        embedding = pca.fit_transform(scale(G))
        results.append(
            {
                "method": "KernelPCA",
                "kernel": "wasserstein_kernel_scaled",
                "kernel_hyperparameter": gamma,
                "ncomponents": k,
                "feature_name": feature_name,
                "embedding": [e.tolist() for e in embedding],
            }
        )

        print("kPCA EWK Histogram")
        pca = KernelPCA(n_components=k, kernel="precomputed", tol=1e-5, max_iter=2000)
        G = pairwise_distances(
            histogram, metric=lambda a, b: eemdkernel(a, b, gamma), n_jobs=-1
        )
        embedding = pca.fit_transform(G)
        results.append(
            {
                "method": "KernelPCA",
                "kernel": "emd_kernel",
                "kernel_hyperparameter": gamma,
                "ncomponents": k,
                "feature_name": feature_name + "_hist",
                "embedding": [e.tolist() for e in embedding],
            }
        )
        embedding = pca.fit_transform(scale(G))
        results.append(
            {
                "method": "KernelPCA",
                "kernel": "emd_kernel",
                "kernel_hyperparameter": gamma,
                "ncomponents": k,
                "feature_name": feature_name + "_hist",
                "embedding": [e.tolist() for e in embedding],
            }
        )

        print("kPCA RBF Histogram")
        D = rbf_kernel(histogram, gamma=gamma)
        embedding = pca.fit_transform(D)
        results.append(
            {
                "method": "KernelPCA",
                "kernel": "rbf_kernel_binning",
                "kernel_hyperparameter": gamma,
                "ncomponents": k,
                "feature_name": feature_name + "_hist",
                "embedding": [e.tolist() for e in embedding],
            }
        )
        embedding = pca.fit_transform(scale(D))
        results.append(
            {
                "method": "KernelPCA",
                "kernel": "rbf_kernel_scaled",
                "kernel_hyperparameter": gamma,
                "ncomponents": k,
                "feature_name": feature_name + "_hist",
                "embedding": [e.tolist() for e in embedding],
            }
        )
    return results


def run_curvatures(json_documents, output, k):
    C = [
        np.array(j["directional_curvature"][2], dtype=np.float64)
        for j in json_documents
    ]
    result = kpca(curvature_histogram(C), C, k, "directional_curvature")
    for f in ["edge_curvature", "node_curvature_edges", "node_curvature_neighborhood"]:
        # C = [np.array(j[f], dtype=np.float64) for j in json_documents]
        C = np.array([j[f] for j in json_documents], dtype=np.float64)
        result += kpca(curvature_histogram(C), C, k, f)
    print(
        json.dumps(result),
        file=(gzip.open(output, "wt") if ".gz" in output else open(output, "wt")),
    )


def run_features(json_documents, output, k):
    result = []
    for f in [
        "edge_cardinality",
        "edge_neighborhood_size",
        "node_degree",
        "node_neighborhood_size",
    ]:
        C = np.array([j[f] for j in json_documents], dtype=np.float64)
        result += kpca(feature_histogram(C), C, k, f)
    print(
        json.dumps(result),
        file=(gzip.open(output, "wt") if ".gz" in output else open(output, "wt")),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input curvature or feature file"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output destination"
    )
    parser.add_argument("--ncomponents", "-k", required=True, type=int)
    args = parser.parse_args()

    if "*" in args.input:
        jo = [
            json.load(gzip.open(i, "r") if ".gz" in i else open(i, "r"))
            for i in glob(args.input)
        ]
    else:
        jo = json.load(
            gzip.open(args.input, "r") if ".gz" in args.input else open(args.input, "r")
        )

    if len(jo) == 0:
        print("Warn: empty file list")
    else:
        if "edge_cardinality" in jo[0].keys():
            run_features(jo, args.output, args.ncomponents)
        else:
            run_curvatures(jo, args.output, args.ncomponents)
