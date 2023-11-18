from kmeans_pytorch import kmeans

import torch
import numpy as np
import time
import torch
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor
from sklearn.cluster import KMeans

use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
device_id = "cuda:0" if use_cuda else "cpu"



def main():
    # data
    data_size, dims1, dims2, num_clusters = 1000, 2, 2, 3
    x = np.random.randn(data_size, dims1, dims2) / 6
    x = torch.from_numpy(x)
    print(x.shape)
    # kmeans
    cluster_ids_x, cluster_centers = kmeans(
        X=x, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
    )
    print("cluster_ids_x")
    print(type(cluster_ids_x), cluster_ids_x.shape)
    print(cluster_ids_x)
    print("cluster_centers")
    print(type(cluster_centers), cluster_centers.shape)
    print(cluster_centers)

def main3():
    data_size, dims1, dims2, num_clusters = 1000, 2, 2, 3
    X = np.random.randn(data_size, dims1, dims2) / 6
    X = torch.from_numpy(X)
    print(X)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

    print(kmeans.labels_)
    print(kmeans.cluster_centers_)


if __name__ == '__main__':
    print("got to try")
    main3()
