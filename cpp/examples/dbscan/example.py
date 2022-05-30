import sys
import argparse
import numpy as np
import cuml
import time
from sklearn.datasets import make_blobs

def create_dataset(n_groups, n_samples_list, n_features, n_clusters, std_dev, rand_state) -> np.ndarray:
    assert(len(n_samples_list) == n_groups)
    dataset = np.zeros(shape=(np.sum(n_samples_list), n_features), dtype=np.float32)
    # start_idx = 0
    # for n_samples in n_samples_list:
    #     end_idx = start_idx + n_samples
    #     dataset[start_idx : end_idx], _ = make_blobs(n_samples=n_samples, n_features=n_features,
    #                                                  centers=n_clusters, cluster_std=std_dev, random_state=rand_state)
    dataset, _ = make_blobs(n_samples=np.sum(n_samples_list), n_features=n_features,
                            centers=n_clusters, cluster_std=std_dev, random_state=rand_state)
    return dataset

arg_parser = argparse.ArgumentParser("example.py")
arg_parser.add_argument('--num_samples_base', type=int, default=20,
                        help='Base number of samples (default 20)')
arg_parser.add_argument('--num_groups', type=int, default=1,
                        help='Number of groups (default 1)')
arg_parser.add_argument('--num_features', type=int, default=32,
                        help='Number of features (default 32)')
arg_parser.add_argument('--min_pts', type=int, default=3,
                        help='Minmum of the number of points (default 3)')
arg_parser.add_argument('--eps', type=float, default=1.0,
                        help='Eps (default 1.0)')
arg_parser.add_argument('--num_clusters', type=int, default=5,
                        help='Number of clusters (default 5)')
arg_parser.add_argument('--standard_dev', type=str, default=0.1,
                        help='Standard deviation of samples generated (default 0.1)')
arg_parser.add_argument('--random_state', type=str, default=123456,
                        help='Standard deviation of samples generated (default 123456)')
args = arg_parser.parse_args()


n_groups        = args.num_groups
n_samples_base  = args.num_samples_base
n_features      = args.num_features
min_pts         = args.min_pts
eps             = args.eps

# Generate dataset
n_samples_array = np.random.randint(0, n_samples_base, size=(n_groups,), dtype=np.int32) + n_samples_base
dataset = create_dataset(n_groups, n_samples_array, n_features, 
                         args.num_clusters, args.standard_dev, args.random_state)
print("=" * 5 + " Create dataset " + "=" * 5)
print(("num_groups = {}\nnum_samples = {}\nnum_features = {}\nnum_clusters = {}").format(
    n_groups, n_samples_array, n_features, args.num_clusters))
print(dataset.shape)

# X = np.random.randn(np.sum(n_samples_array), n_features).astype(np.float32) * 3.0 - 1.5

def run_dbscan_single_batch(X, eps, min_pts):
    dbscan = cuml.DBSCAN(eps=eps, min_samples=min_pts)
    dbscan.fit(X)
    print(dbscan.labels_)
    return dbscan.labels_

def run_dbscan_multi_batch(X, n_samples_array, eps, min_pts):
    dbscan_batch = cuml.DBSCAN(eps=eps, min_samples=min_pts)
    dbscan_batch.fit_batched(X, n_samples_array, out_dtype=n_samples_array.dtype)
    print(dbscan_batch.labels_)
    return dbscan_batch.labels_

print("=" * 5 + " Batch version " + "=" * 5)
start_time = time.time()
label_multi_batch = run_dbscan_multi_batch(dataset, n_samples_array, eps, min_pts)
time_multi = time.time() - start_time

print("=" * 5 + " Compare version " + "=" * 5)
label_single_tmp = list()
time_single = 0.0
start_idx = 0
for n_samples in n_samples_array:
    start_time = time.time()
    end_idx = start_idx + n_samples
    data_epoch = dataset[start_idx : end_idx]
    tmp = run_dbscan_single_batch(data_epoch, eps, min_pts)
    time_single += time.time() - start_time
    label_single_tmp.append(tmp)
    start_idx = end_idx
label_single_batch = np.concatenate(label_single_tmp)

print("Diff =", np.sum(np.abs(label_multi_batch - label_single_batch)))
print("Time multi = {} single = {} speedup = {}".format(time_multi, time_single, time_single / time_multi))
