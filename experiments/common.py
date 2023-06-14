import math

import numpy as np
import ot
import pymp
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances

# from multiprocessing import Pool
# def pmap(f, x, num_threads):
#     with Pool(processes=num_threads) as p:
#         p.map(f, x)


def eemdkernel(x, y, gamma):
    return math.exp(-gamma * ot.lp.emd2_1d(x, y))


def pmap(f, x, num_threads):
    y = pymp.shared.list([None] * len(x))
    with pymp.Parallel(num_threads) as p:
        for i in p.range(len(x)):
            y[i] = f(x[i])
    return list(y)


def wasserstein_kernel(x, y, gamma):
    return math.exp(-gamma * wasserstein_distance(x, y))


def wasserstein_cluster_coeffcient(C, labels, n_jobs=1):
    L = np.unique(labels)
    intra, inter = np.zeros(len(L)), np.zeros((len(L), len(L)))
    for i in range(len(L)):
        Ki = pairwise_distances(
            C[labels == L[i]],
            metric=lambda a, b: wasserstein_distance(a, b),
            n_jobs=n_jobs,
        )
        intra[i] = Ki.sum() * 2 / max(len(Ki) * (len(Ki) - 1), 1)
        for j in range(i + 1, len(L)):
            Kij = pairwise_distances(
                C[labels == L[i]],
                Y=C[labels == L[j]],
                metric=lambda a, b: wasserstein_distance(a, b),
                n_jobs=n_jobs,
            )
            inter[i][j] = Kij.sum() / np.prod(Kij.shape)
    wcc = intra.sum() / (1 + inter.sum())
    return wcc, intra, inter


def _binning(C, bins):
    def H(c):
        h = np.histogram(c, bins)[0]
        return h / sum(h)

    return list(pmap(H, C, 64))


def curvature_histogram(C):
    return _binning(C, [x / 100 for x in range(-200, 104, 5)])


def feature_histogram(C):
    x = [x for c in C for x in c]
    return _binning(C, np.linspace(np.min(x), np.max(x), 61))
