import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl
from scipy.cluster import hierarchy
from sklearn.datasets import make_blobs

""" Ward' method bottom up clustering."""

mpl.use('macosx')


if __name__ == '__main__':
    X, _ = make_blobs(n_samples=2000, n_features=3, centers=8)
    print(hierarchy.linkage(X, method='ward'))
    dendrogram = hierarchy.dendrogram(hierarchy.linkage(X, method='ward'))
    plt.title('Dendrogram')
    plt.xlabel('Elements')
    plt.ylabel('Euclidean distances')
    plt.show()
