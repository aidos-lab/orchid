import argparse
import gzip
import json
from glob import glob
from itertools import combinations

import numpy as np
from common import *
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


def mmd(x, y, sigma):
    n = len(x)
    xy = np.concatenate((x, y))
    K = np.exp((-1 / (2 * sigma**2)) * np.power(pairwise_distances(xy, xy), 2))
    XX, YY, XY = K[:n, :n], K[n:, n:], K[:n, n:]
    return XX.mean() + YY.mean() - 2 * XY.mean()


def bootstrapping(x, y, fn, iterations=100):
    n = len(x)
    xy = np.concatenate((x, y))
    b = []
    for _ in tqdm(range(iterations)):  # tqdm
        xy = xy[np.random.permutation(len(xy))]
        b.append(fn(xy[:n], xy[n:]))
    return np.array(b)


def mmd_emd(C, feature_type):
    def emd_bootstrapping(t):
        i, j = t
        m = wasserstein_distance(C[i], C[j])
        t = bootstrapping(C[i], C[j], lambda a, b: wasserstein_distance(a, b))
        return i, j, m, t, (t >= m).mean()

    print("Bootstrapping EMD on curvatures")
    R = pmap(emd_bootstrapping, [t for t in combinations(range(len(C)), 2)], 16)
    I = [r[0] for r in R]
    J = [r[1] for r in R]
    M = [r[2] for r in R]
    P = [r[4] for r in R]
    return {
        "feature_type": feature_type,
        "gamma": 1,
        "metric_type": "wasserstein_distance",
        "values": {"I": I, "J": J, "K": M, "pvalues": P},
    }


def mmd_experiments(C, feature_type, bins, gammas=[1]):
    def hist(c):
        h = np.histogram(c, bins)[0]
        return h / sum(h)

    result = [mmd_emd(C, feature_type)]
    for gamma in gammas:

        def mmd_bootstrapping(t):
            i, j = t
            m = mmd(hist(C[i]).reshape(-1, 1), hist(C[j]).reshape(-1, 1), gamma)
            t = bootstrapping(
                C[i],
                C[j],
                lambda a, b: mmd(hist(a).reshape(-1, 1), hist(b).reshape(-1, 1), gamma),
            )
            return i, j, m, t, (t >= m).mean()

        print("Bootstrapping MMD on histograms")
        R = pmap(mmd_bootstrapping, [t for t in combinations(range(len(C)), 2)], 16)
        I = [r[0] for r in R]
        J = [r[1] for r in R]
        M = [r[2] for r in R]
        P = [r[4] for r in R]
        result.append(
            {
                "feature_type": feature_type,
                "gamma": gamma,
                "metric_type": "mmd",
                "values": {"I": I, "J": J, "K": M, "pvalues": P},
            }
        )

    return result


def run_curvatures(json_documents, output):
    bins = [x / 100 for x in range(-200, 104, 5)]
    C = [
        np.array(j["directional_curvature"][2], dtype=np.float64)
        for j in json_documents
    ]
    result = mmd_experiments(C, "directional_curvature", bins)
    for f in ["edge_curvature", "node_curvature_edges", "node_curvature_neighborhood"]:
        C = [np.array(j[f], dtype=np.float64) for j in json_documents]
        result += mmd_experiments(C, f, bins)
    print(
        json.dumps(result),
        file=(gzip.open(output, "wt") if ".gz" in output else open(output, "wt")),
    )


def run_features(json_documents, output):
    result = []
    for f in [
        "edge_cardinality",
        "edge_neighborhood_size",
        "node_degree",
        "node_neighborhood_size",
    ]:
        C = [np.array(j[f], dtype=np.float64) for j in json_documents]
        lb = np.min([x for c in C for x in c])
        ub = np.max([x for c in C for x in c])
        bins = np.linspace(lb, ub, 61)
        result += mmd_experiments(C, f, bins)
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
            run_features(jo, args.output)
        else:
            run_curvatures(jo, args.output)
