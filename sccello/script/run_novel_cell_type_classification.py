import os
import sys
import pickle
import argparse
from tqdm import tqdm
import pandas as pd
import scipy
import ipdb
import numpy as np
import networkx as nx
import logging
from sklearn.metrics import accuracy_score, f1_score

import torch

EXC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(EXC_DIR)

from sccello.src import cell_ontology
from sccello.src.utils import helpers, data_loading, logging_util
from sccello.src.data.dataset import CellTypeClassificationDataset

logging.basicConfig(level=logging.INFO)

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", type=str, default="novel_cell_type_classification", 
                            choices=["novel_cell_type_classification"])
    parser.add_argument("--pretrained_ckpt", type=str, help="pretrained model checkpoints", required=True)
    parser.add_argument("--target_data_mode", type=str, default="celltype_data1", 
                        choices=["celltype_data1", "celltype_data2"])
    parser.add_argument("--pretrain_data_branch", type=str, default="frac100")
    parser.add_argument("--output_dir", type=str, default="./single_cell_output", required=True)
    parser.add_argument("--model_source", type=str, default="model_prototype_contrastive")
    parser.add_argument("--wandb_run_name", help="wandb run name", type=str, 
                        default="test", required=True)
    parser.add_argument("--indist_repr_path", type=str, required=True)
    
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--ckpt_num", type=str, default="")
    
    args = parser.parse_args()
    
    # NOTE
    args.method = args.pretrained_ckpt.strip("/").split("/")[-2]
    args.ckpt_num = args.pretrained_ckpt.strip("/").split("/")[-1].split("-")[1]

    args.model_class = {
        "model_prototype_contrastive": "PrototypeContrastiveForMaskedLM",
    }[args.model_source]

    assert os.path.exists(args.indist_repr_path), "D^{id} data not inferenced. Run indist=1 for cell type clustering"
    
    return args

def get_cell_type_labelid2nodeid(cell_type_idmap, clid2nodeid):
    
    # e.g., {'CL:2000078': 'Placental pericyte', 'CL:2000054': 'Hepatic pit cell', ...}
    clid2name = data_loading.get_prestored_data("cell_taxonomy_tree_json")[-1]

    # e.g., {0: 'connective tissue cell', 1: 'myeloid dendritic cell', ...}
    cell_type2name = {v: k.lower() for k, v in cell_type_idmap.items()}

    # e.g., {'placental pericyte': CL:2000078, ...}
    name2clid = {v.lower(): k for k, v in clid2name.items()}

    cell_type2nodeid = dict([(k, clid2nodeid[name2clid[cell_type2name[k]]]) if cell_type2name[k] in name2clid else (k, -1) for k in cell_type2name])
    return cell_type2nodeid

def load_cell_type_representation(args, model):

    h_eval = pickle.load(open(args.indist_repr_path, "rb"))
    h_eval, tag = helpers.cellrepr_post_process(model, h_eval, pass_cell_cls=1, normalize=False)

    all_datasets = data_loading.get_fracdata("celltype", "frac100", True, False)
    _, evalset, _, _ = all_datasets # a subsample of training data
    labels = evalset["label"]
    labels = torch.tensor(labels)

    n_labels, n_dim = max(labels) + 1, h_eval.shape[1]
    celltype_repr = torch.zeros(n_labels, n_dim)
    celltype_repr.scatter_reduce_(0, labels.unsqueeze(1).expand(-1, n_dim), torch.tensor(h_eval), reduce="mean", include_self=False)

    return celltype_repr
    

def load_cell_representation(args, model):
    file_name = os.path.join(EXC_DIR, f"./embedding_storage/cellreprs_frac_{args.target_data_mode}.pkl")
    h_data = pickle.load(open(file_name, "rb"))

    h_data, tag = helpers.cellrepr_post_process(model, h_data, pass_cell_cls=1, normalize=False)
    print("cell_repr.shape: ", h_data.shape)
    return torch.tensor(h_data)
    

def get_unified_mapping_and_similarity(pretrain_label_map, target_label_map):
    """
    Create unified ID mapping and compute voting similarity.
    """
    pretrain_label_len, target_label_len = len(pretrain_label_map), len(target_label_map)
    ct_graph_edge_list, ct_graph_vocab_clid2idx = data_loading.get_prestored_data("cell_taxonomy_graph_owl")
    
    # get graph similarity
    ct_graph = nx.Graph(ct_graph_edge_list)
    similarity = cell_ontology.get_cell_taxonomy_similarity(ct_graph, get_raw=True)
    
    # get merged idmap
    pretrain_idmap = get_cell_type_labelid2nodeid(pretrain_label_map,  ct_graph_vocab_clid2idx)
    target_idmap = get_cell_type_labelid2nodeid(target_label_map, ct_graph_vocab_clid2idx)
    final_idmap = target_idmap
    for k, v in pretrain_idmap.items():
        final_idmap[k + target_label_len] = v

    # extract voting similarity from graph similarity
    voting_sim = np.zeros((target_label_len, pretrain_label_len))
    for i in range(target_label_len):
        for j in range(pretrain_label_len):
            mapped_i = final_idmap[i]
            mapped_j = final_idmap[target_label_len + j]
            if mapped_i != -1 and mapped_j != -1:
                voting_sim[i][j] = similarity[mapped_i][mapped_j]
            else:
                voting_sim[i][j] = 0
    
    return final_idmap, voting_sim

def evaluate_predictions(args, cls_probs, cls_pred, celltype_labels, target_label_map):
    """Results for Figure 2 in scCello paper."""

    N = 20 # times of sampling
    num_cell_types = len(target_label_map)
    all_result, avg_result = [], []
    for ratio in [0.1, 0.25, 0.5, 0.75, 1.0]:
        num_selected_cell_types = int(num_cell_types * ratio + 0.5)

        all_acc, all_f1 = [], []
        for _ in range(N):
            selected_celltypes = np.random.permutation(num_cell_types)[:num_selected_cell_types]
            mapping = {c:idx for idx,c in enumerate(selected_celltypes)}
            
            used_indices = np.isin(celltype_labels, selected_celltypes)
            new_labels = [mapping[x] for x in celltype_labels[used_indices]]
            new_preds = np.argmax(np.array(cls_probs)[used_indices][:, selected_celltypes], axis=-1)

            acc = accuracy_score(new_labels, new_preds)
            f1 = f1_score(new_labels, new_preds, average="macro")
            all_acc.append(acc)
            all_f1.append(f1)
            if ratio >= 1.0: # same results for each run
                break
        
        divnum = 1 if ratio >= 1.0 else N
        avg_result.append((round(sum(all_acc) / divnum, 4), round(sum(all_f1) / divnum, 4)))
        all_result.append((all_acc, all_f1))

    avg_result = pd.DataFrame(np.array(avg_result), columns=["acc", "f1"])
    avg_result.to_csv(f"novel_cell_type_classification_{args.target_data_mode}.csv", index=False)

    print("# --------- Overall ---------")
    print("acc: ", accuracy_score(celltype_labels, cls_pred))
    print("f1: ", f1_score(celltype_labels, cls_pred, average="macro"))

if __name__ == "__main__":
    args = parse_args()
    logging_util.set_environ(args)

    pretrain_label_map = data_loading.get_prestored_data(f"pretrain_newdata_{args.pretrain_data_branch}_cell_type_idmap_pkl")
    # target data is D^{ct}_1 or D^{ct}_2
    data1, data2 = CellTypeClassificationDataset.create_dataset("celltype")
    target_set = data1 if "data1" in args.target_data_mode else data2
    celltype_labels, target_label_map = helpers.labels_category2int(target_set["label"], return_map=True)

    # load the pre-trained model
    model = helpers.load_model_inference(args)

    h_celltype = load_cell_type_representation(args, model)
    h_data = load_cell_representation(args, model)
    assert len(h_data) == len(celltype_labels)

    final_idmap, voting_sim = get_unified_mapping_and_similarity(pretrain_label_map, target_label_map)

    cls_pred, cls_probs = [], []
    for idx, cell_type in enumerate(tqdm(celltype_labels)):
        
        h_x = torch.nn.functional.normalize(h_data[idx].unsqueeze(0), p=2, dim=-1) / 0.1
        h_y = torch.nn.functional.normalize(h_celltype, p=2, dim=-1) / 0.1
        sim_x = (h_x * h_y).sum(-1).detach().numpy()
        logits = torch.tensor(scipy.stats.spearmanr(sim_x, voting_sim, axis=1)[0][0][1:])
        
        cls_pred.append(torch.argmax(logits).item())
        cls_probs.append(torch.nn.functional.softmax(logits).detach().numpy())

    evaluate_predictions(args, cls_probs, cls_pred, celltype_labels, target_label_map)
    