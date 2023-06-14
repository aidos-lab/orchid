# pip install git+https://github.com/alan-turing-institute/SigNet.git
# doi.org/10.5281/zenodo.1435036
import argparse
import gzip
import json

import numpy as np
from common import *
from scipy.sparse import csr_matrix
from signet.cluster import Cluster
from sklearn.cluster import SpectralClustering


def exp_kernel(W, gamma):
    return np.exp(-gamma * W)


def sncluster(sn, k, gamma, feature_name, kernel_name):
    result = []
    y = sn.spectral_cluster_adjacency_reg(k=k)
    result.append(
        {
            "method": "spectral_cluster_adjacency_reg",
            "kernel": kernel_name,
            "feature": feature_name,
            "ncomponents": k,
            "gamma": gamma,
            "labels": y.tolist(),
        }
    )
    y = sn.spectral_cluster_bnc(k=k)
    result.append(
        {
            "method": "spectral_cluster_bnc",
            "kernel": kernel_name,
            "feature": feature_name,
            "ncomponents": k,
            "gamma": gamma,
            "labels": y.tolist(),
        }
    )
    y = sn.spectral_cluster_laplacian(k=k)
    result.append(
        {
            "method": "spectral_cluster_laplacian",
            "kernel": kernel_name,
            "feature": feature_name,
            "ncomponents": k,
            "gamma": gamma,
            "labels": y.tolist(),
        }
    )
    # y = sn.geproblem_adjacency(k=k)
    # result.append({"method": "geproblem_adjacency", "kernel": kernel_name, "feature": feature_name, "ncomponents": k, "gamma": gamma, "labels": y.tolist()})
    y = sn.SPONGE(k=k)
    result.append(
        {
            "method": "SPONGE",
            "kernel": kernel_name,
            "feature": feature_name,
            "ncomponents": k,
            "gamma": gamma,
            "labels": y.tolist(),
        }
    )
    y = sn.SPONGE_sym(k=k)
    result.append(
        {
            "method": "SPONGE_sym",
            "kernel": kernel_name,
            "feature": feature_name,
            "ncomponents": k,
            "gamma": gamma,
            "labels": y.tolist(),
        }
    )
    # y = sn.spectral_cluster_bethe_hessian(k=k)
    # result.append({"method": "spectral_cluster_bethe_hessian", "kernel": kernel_name, "feature": feature_name, "ncomponents": k, "gamma": gamma, "labels": y.tolist()})
    y = sn.SDP_cluster(k=k, normalisation="none")
    result.append(
        {
            "method": "SDP_cluster",
            "kernel": kernel_name,
            "feature": feature_name,
            "ncomponents": k,
            "gamma": gamma,
            "labels": y.tolist(),
        }
    )
    return result


def run_node_clustering(C, k):
    spec = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        n_jobs=-1,
        eigen_solver="lobpcg",
        eigen_tol=1e-2,
        assign_labels="cluster_qr",
    )
    result = []

    W = 1 - C
    for gamma in [0.5, 1.0, 2.0, 1 / W.std()]:
        spec = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            n_jobs=-1,
            eigen_solver="lobpcg",
            eigen_tol=1e-2,
            assign_labels="cluster_qr",
        )
        y = spec.fit_predict(exp_kernel(W, gamma))
        result.append(
            {
                "method": "SpectralClustering",
                "kernel": "wasserstein_kernel",
                "feature": "directional_wasserstein_distance",
                "ncomponents": k,
                "gamma": gamma,
                "labels": y.tolist(),
            }
        )
    y = spec.fit_predict(C)
    result.append(
        {
            "method": "SpectralClustering",
            "kernel": "precomputed",
            "feature": "directional_curvature",
            "ncomponents": k,
            "labels": y.tolist(),
        }
    )
    # y = spec.fit_predict(-C)
    # result.append(
    #     {
    #         "method": "SpectralClustering",
    #         "kernel": "precomputed_inverted",
    #         "feature": "directional_curvature",
    #         "ncomponents": k,
    #         "labels": y.tolist(),
    #     }
    # )
    D = pairwise_distances(C)
    for gamma in [1.0, 1 / D.std()]:
        Kern = exp_kernel(D, gamma)
        spec = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            n_jobs=-1,
            eigen_solver="lobpcg",
            eigen_tol=1e-2,
            assign_labels="cluster_qr",
        )
        y = spec.fit_predict(Kern)
        result.append(
            {
                "method": "SpectralClustering",
                "kernel": "exp_kernel",
                "feature": "directional_curvature",
                "ncomponents": k,
                "gamma": gamma,
                "labels": y.tolist(),
            }
        )
        y = spec.fit_predict((C != 0) * Kern)
        result.append(
            {
                "method": "SpectralClustering",
                "kernel": "exp_kernel_adj",
                "feature": "directional_curvature",
                "ncomponents": k,
                "gamma": gamma,
                "labels": y.tolist(),
            }
        )

    y = spec.fit_predict(1.0 * (C != 0))
    result.append(
        {
            "method": "SpectralClustering",
            "kernel": "unweighted_adj",
            "feature": "directional_curvature",
            "ncomponents": k,
            "gamma": gamma,
            "labels": y.tolist(),
        }
    )
    result += sncluster(
        Cluster((csr_matrix((1.0 * (C > 0))), csr_matrix(1.0 * (C < 0)))),
        k,
        1,
        "directional_curvature",
        "unweighted_signed_adj",
    )
    result += sncluster(
        Cluster(
            (
                csr_matrix(np.where(C > 0, C, 0 * C)),
                csr_matrix(np.where(C < 0, -C, 0 * C)),
            )
        ),
        k,
        1,
        "directional_curvature",
        "weighted_signed_adj"
        # Cluster((C * (C > 0), -C * (C < 0))), k, 1, feature_name, "weighted_signed_adj"
    )
    return result


def run_curvatures(json_document, output, k):
    IJK = json_document["directional_curvature"]
    K, I, J = (
        np.array(IJK[2], dtype=np.float64),
        np.array(IJK[0], dtype=np.int64),
        np.array(IJK[1], dtype=np.int64),
    )
    n = max(I.max(), J.max())
    C = csr_matrix((K, (I - 1, J - 1)), shape=(n, n)).toarray()
    C = C + C.T
    np.fill_diagonal(C, 1)
    # C = np.array(json_document["directional_curvature"][2], dtype=np.float64)
    result = run_node_clustering(C, k)
    for f in ["node_curvature_edges", "node_curvature_neighborhood"]:
        C = np.array(json_document[f], dtype=np.float64).reshape(-1, 1)
        D = pairwise_distances(C)
        for gamma in [1.0, 1 / D.std()]:
            Kern = exp_kernel(D, gamma)
            spec = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                n_jobs=-1,
                eigen_solver="lobpcg",
                eigen_tol=1e-2,
                assign_labels="cluster_qr",
            )
            y = spec.fit_predict(Kern)
            result.append(
                {
                    "method": "SpectralClustering",
                    "kernel": "rbfkernel",
                    "feature": f,
                    "ncomponents": k,
                    "gamma": gamma,
                    "labels": y.tolist(),
                }
            )
            y = spec.fit_predict((C != 0) * Kern)
            result.append(
                {
                    "method": "SpectralClustering",
                    "kernel": "rbfkernel",
                    "feature": f"adj-{f}",
                    "ncomponents": k,
                    "gamma": gamma,
                    "labels": y.tolist(),
                }
            )
    print(
        json.dumps(result),
        file=(gzip.open(output, "wt") if ".gz" in output else open(output, "wt")),
    )


def run_features(json_document, output, k):
    result = []
    for feature in ["node_degree", "node_neighborhood_size"]:
        d = np.array(json_document[feature]).reshape(-1, 1)
        D = pairwise_distances(d)
        for gamma in [0.5, 1.0, 2.0, 1 / D.std()]:
            spec = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                n_jobs=-1,
                eigen_solver="lobpcg",
                eigen_tol=1e-2,
                assign_labels="cluster_qr",
            )
            y = spec.fit_predict(exp_kernel(D, gamma))
            result.append(
                {
                    "method": "SpectralClustering",
                    "kernel": "rbfkernel",
                    "feature": feature,
                    "ncomponents": k,
                    "gamma": gamma,
                    "labels": y.tolist(),
                }
            )
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

    jo = json.load(
        gzip.open(args.input, "r") if ".gz" in args.input else open(args.input, "r")
    )

    if len(jo) == 0:
        print("Warn: empty file list")
    else:
        if "edge_cardinality" in jo.keys():
            run_features(jo, args.output, args.ncomponents)
        else:
            run_curvatures(jo, args.output, args.ncomponents)
