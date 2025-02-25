import os
import time
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scib
import ipdb

import torch
# assert torch.cuda.is_available()

import cupy as cp
from cuml.metrics.cluster import silhouette_score as cu_silhouette_score
from cuml.metrics.cluster import silhouette_samples as cu_silhouette_samples
import rmm

from sccello.src.utils import helpers

EXC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def squash_array_of_metrics(log_infos):
    """
    merge a list of dict into one dict, with each metrics averaged
    """
    num_rounds = len(log_infos)
    log_infos_mean, log_infos_std = {}, {}
    for key in log_infos[0].keys():
        tmp = [log_infos[round_idx][key] for round_idx in range(num_rounds)]
        mean_val, std_val = np.array(tmp).mean(), np.array(tmp).std()
        log_infos_mean[key], log_infos_std[key] = mean_val, std_val
    
    return log_infos_mean, log_infos_std

def eval_scanpy_gpu_metrics_solver(
    adata, label_key="cell_type", cluster_key="cluster", embedding_key="X_embed", 
    verbose=0, use_anndata=True, batch_effect=False
):
    rmm.reinitialize(managed_memory=True, # Allows oversubscription
        devices=0, # GPU device IDs to register. By default registers only GPU 0.
    )
    cp.cuda.set_allocator(rmm.allocators.cupy.rmm_cupy_allocator)
    if not use_anndata:
        h_repr, labels = adata
        if not isinstance(labels[0], int):
            labels = helpers.labels_category2int(labels)
        adata = anndata.AnnData(h_repr, obs=pd.DataFrame({"cell_type": np.array(labels)}))
        adata.obsm[embedding_key] = adata.X
    else:
        if cluster_key in adata.obs.columns:
            print(f"Warning: cluster key {cluster_key} already exists "
                "in adata.obs and will be overwritten")
    
    # pre-clustering with rapids on GPU for acceleration
    def func_clustering(adata, inplace=True):
        preprocessing_start = time.time()
        n = 20 # default practice https://github.com/theislab/scib/blob/ed3e2846414ca1e3dc07552c0eef1e68d82230d4/scib/metrics/clustering.py#L130
        resolutions = [2 * x / n for x in range(1, n + 1)]
        score_max, res_max, clustering, score_all = 0, resolutions[0], None, []
        sc.pp.neighbors(adata, use_rep=embedding_key, method='rapids')
        for res in resolutions:
            sc.tl.louvain(adata, resolution=res, key_added=cluster_key, flavor='rapids')
            score = scib.metrics.nmi(adata, label_key, cluster_key)
            score_all.append(score)
            if score_max < score:
                score_max = score
                res_max = res
                clustering = adata.obs[cluster_key]
            del adata.obs[cluster_key]
        if verbose:
            print(f"optimised clustering against {label_key}")
            print(f"optimal cluster resolution: {res_max}")
            print(f"optimal score: {score_max}")
        score_all = pd.DataFrame(
            zip(resolutions, score_all), columns=("resolution", "score")
        )
        if inplace:
            adata.obs[cluster_key] = clustering
        if verbose:
            print("[func_clustering]: ", time.time() - preprocessing_start)
        return res_max, score_max, score_all
    
    results_dict = dict()
    if not batch_effect:
        res_max, nmi_max, nmi_all = func_clustering(adata, inplace=True)
        results_dict["NMI_cluster/label"] = scib.metrics.nmi(
            adata, cluster_key, label_key, "arithmetic", nmi_dir=None
        )
        results_dict["ARI_cluster/label"] = scib.metrics.ari(
            adata, cluster_key, label_key
        )
    def func_asw_cu(adata, variant=False):
        # see scCello manuscript for App.D.1: Average Silhouette Width Score and Silhouette Variant Score
        preprocessing_start = time.time()
        if not variant:
            asw = cu_silhouette_score(adata.obsm[embedding_key], labels=adata.obs[label_key], metric="euclidean")
        else:
            all_asw = cu_silhouette_samples(adata.obsm[embedding_key], labels=adata.obs[label_key], metric="euclidean")
            all_asw = np.abs(all_asw)
            label_ids = adata.obs[label_key].to_numpy()
            asw = []
            for i in range(label_ids.max() + 1):
                if (label_ids == i).any():
                    asw.append(1 - np.mean(all_asw[label_ids == i]).item())
            asw = np.mean(asw)

        if verbose:
            print("[func_asw_cu]: ", time.time() - preprocessing_start)
        return (asw + 1) / 2.0
    
    if batch_effect:
        results_dict["ASW_batch"] = func_asw_cu(adata, variant=True)
        sc.pp.neighbors(adata, use_rep=embedding_key, method='rapids')
        adata.obs[f"{label_key}_str"] =  adata.obs[label_key].astype("category")
        results_dict["graph_conn"] = scib.metrics.graph_connectivity(adata, f"{label_key}_str")
        results_dict["avg_batch"] = np.mean([
            results_dict["graph_conn"],
            results_dict["ASW_batch"],]
        )
    else:
        results_dict["ASW_label"] = func_asw_cu(adata)
        results_dict["avg_bio"] = np.mean([
            results_dict["NMI_cluster/label"],
            results_dict["ARI_cluster/label"],
            results_dict["ASW_label"]]
        )
    
    # remove nan value in result_dict
    results_dict = {k: v for k, v in results_dict.items() if not np.isnan(v)}
    return results_dict

def eval_scanpy_gpu_metrics(adata, fast_mode=False, use_anndata=True, fast_ratio=None, **kwargs):
    
    preprocessing_start = time.time()
    if use_anndata:
        num_cells = adata.n_obs
    else:
        num_cells = len(adata[0])
    
    # random sampling if the data is too large for fast evaluation
    if num_cells > 500000 or fast_mode:
        N = 10
        if fast_ratio is None:
            subset_size = int(num_cells / N)
        else:
            subset_size = int(num_cells * fast_ratio)
            N = N ** 2

        all_scores = []
        for _ in range(N):
            selected_indices = np.random.permutation(num_cells)[:subset_size]

            if use_anndata:
                subset_adata = adata[selected_indices].copy()
            else:
                h_repr, labels = adata
                subset_h_repr = h_repr[selected_indices]
                try:
                    subset_labels = np.array(labels)[selected_indices].tolist()
                except:
                    ipdb.set_trace()
                subset_adata = (subset_h_repr, subset_labels)
            
            x = eval_scanpy_gpu_metrics_solver(subset_adata, use_anndata=use_anndata, **kwargs)
            all_scores.append(x)
        
        all_scores = squash_array_of_metrics(all_scores)
        if kwargs["verbose"]:
            print("std: ", all_scores[1])
        all_scores = all_scores[0]
    else:
        all_scores = eval_scanpy_gpu_metrics_solver(adata, use_anndata=use_anndata, **kwargs)
    
    if kwargs["verbose"]:
        print("\n".join([f"{k}: {v:.4f}" for k, v in all_scores.items()]))
        print("[eval_scanpy_gpu_metrics]: ", time.time() - preprocessing_start)
    return all_scores

