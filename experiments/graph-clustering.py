import argparse
import gzip
import json
from glob import glob

import numpy as np
from common import *
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel


def clustering(histogram, values, k, feature_name, gammas=[1]):
    results = []
    for gamma in gammas:
        spec = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            n_jobs=-1,
            eigen_solver="lobpcg",
            eigen_tol=1e-10,
            assign_labels="cluster_qr",
            # assign_labels="kmeans",
        )

        print("Spectral Clustering EWK")
        G = pairwise_distances(
            values, metric=lambda a, b: wasserstein_kernel(a, b, gamma), n_jobs=-1
        )
        y = spec.fit_predict(G)
        wcc, intra, inter = wasserstein_cluster_coeffcient(
            np.array(values), np.array(y), n_jobs=128
        )
        # print("EWK", feature_name, wcc, y)
        results.append(
            {
                "method": "spectral_clustering",
                "kernel": "wasserstein_kernel",
                "kernel_hyperparameter": gamma,
                "ncomponents": k,
                "feature_name": feature_name,
                "labels": y.tolist(),
                "wasserstein_clustering_coefficient": {
                    "total": wcc,
                    "internal": intra.tolist(),
                    "external": inter.tolist(),
                },
            }
        )

        print("Spectral Clustering EWK Histogram")
        G = pairwise_distances(
            histogram, metric=lambda a, b: eemdkernel(a, b, gamma), n_jobs=-1
        )
        y = spec.fit_predict(G)
        wcc, intra, inter = wasserstein_cluster_coeffcient(
            np.array(values), np.array(y), n_jobs=128
        )
        # print("EWK", feature_name, wcc, y)
        results.append(
            {
                "method": "spectral_clustering",
                "kernel": "emd_kernel",
                "kernel_hyperparameter": gamma,
                "ncomponents": k,
                "feature_name": feature_name + "_hist",
                "labels": y.tolist(),
                "wasserstein_clustering_coefficient": {
                    "total": wcc,
                    "internal": intra.tolist(),
                    "external": inter.tolist(),
                },
            }
        )

        print("Spectral Clustering RBF Histogram")
        G = rbf_kernel(histogram, gamma=gamma)
        y = spec.fit_predict(G)
        wcc, intra, inter = wasserstein_cluster_coeffcient(
            np.array(values), np.array(y), n_jobs=128
        )
        # print("RBF", feature_name + "_hist", wcc, y)
        results.append(
            {
                "method": "spectral_clustering",
                "kernel": "rbf_kernel",
                "kernel_hyperparameter": gamma,
                "ncomponents": k,
                "feature_name": feature_name + "_hist",
                "labels": y.tolist(),
                "wasserstein_clustering_coefficient": {
                    "total": wcc,
                    "internal": intra.tolist(),
                    "external": inter.tolist(),
                },
            }
        )

    return results


def run_curvatures(json_documents, output, k):
    C = [
        np.array(j["directional_curvature"][2], dtype=np.float64)
        for j in json_documents
    ]
    result = clustering(curvature_histogram(C), C, k, "directional_curvature")
    for f in ["edge_curvature", "node_curvature_edges", "node_curvature_neighborhood"]:
        C = [np.array(j[f], dtype=np.float64) for j in json_documents]
        result += clustering(curvature_histogram(C), C, k, f)
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
        C = [np.array(j[f], dtype=np.float64) for j in json_documents]
        result += clustering(feature_histogram(C), C, k, f)
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
