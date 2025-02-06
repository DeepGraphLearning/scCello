import os
import sys
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import logging

import torch

EXC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(EXC_DIR)

from sccello.src.utils import helpers, logging_util, data_loading
from sccello.src.data.dataset import MarkerGenePredictionDataset

logging.basicConfig(level=logging.INFO)

def parse_args():

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--objective", default="marker_gene_prediction",
                        choices=["marker_gene_prediction"])
    parser.add_argument("--pretrained_ckpt", type=str, help="pretrained model checkpoints",)
    parser.add_argument("--output_dir", type=str, default="./single_cell_output", required=True)
    parser.add_argument("--seed", help="random seed", type=int, default=42)
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_source", type=str, default="model_prototype_contrastive")

    parser.add_argument("--wandb_project", help="wandb project name", type=str, default="marker_gene_prediction")
    parser.add_argument("--wandb_run_name", help="wandb run name", type=str, 
                        default="test", required=True)
    
    args = parser.parse_args()

    if args.pretrained_ckpt.endswith("/"):
        args.pretrained_ckpt = args.pretrained_ckpt[:-1]
    
    args.model_class = {
        "model_prototype_contrastive": "PrototypeContrastiveForMaskedLM",
    }[args.model_source]
    
    print("args: ", args)

    return args

def get_perturbed_logits(args, model, input_ids, label_dict, cell_label2marker):

    mask_token_id = data_loading.get_prestored_data("token_dictionary_pkl")["<mask>"]

    perturbed_input_ids = torch.tensor(input_ids).to("cuda") # [L, ] L includes <cls> as the first position
    perturbed_input_ids = perturbed_input_ids.repeat(len(input_ids), 1) # [L, L]
    perturbed_input_ids[1:, 1:].fill_diagonal_(mask_token_id) # the first cell is the original cell

    # input of the same length, no need of calling Trainer for batching and padding
    perturbed_logits = []
    for i in range(0, len(perturbed_input_ids), args.batch_size):
        output = model(perturbed_input_ids[i: i + args.batch_size], )
        logits = output.logits # [bsz, num_layers + 1, dim]
        logits = logits[:, -1, :] # [bsz, dim]
        perturbed_logits.append(logits)
    perturbed_logits = torch.cat(perturbed_logits, dim=0) # [L, dim]
    perturbed_logits = torch.nn.functional.normalize(perturbed_logits, dim=-1)
    
    return perturbed_logits

def get_cell_type_label2marker(data_name):

    cell_markers_tsv = data_loading.get_prestored_data("cell_markers_tsv")
    cell_type2markers = cell_markers_tsv.set_index("cellType").to_dict()["geneMarker"]
    token_vocab_dict = data_loading.get_prestored_data("token_dictionary_pkl")

    cell_label2types_csv = data_loading.get_prestored_data("cell_label2types_csv")

    cell_label2marker = {}
    for x in cell_label2types_csv.iloc():
        data, original_label, aligned_label = x["dataset"], x["original label"], x["aligned label"]
        if data in data_name:
            cell_label2marker[original_label] = []
            for aligned_type in aligned_label.split(","):
                if aligned_type not in cell_type2markers:
                    continue
                marker_list = cell_type2markers[aligned_type].split(",")
                marker_ensembl_ids = data_loading.map_gene_name2id(np.array(marker_list))
                marker_input_ids = [token_vocab_dict[x] for x in marker_ensembl_ids if x is not None and x != 'None']
                cell_label2marker[original_label] += marker_input_ids
            cell_label2marker[original_label] = list(set(cell_label2marker[original_label]))

    return cell_label2marker

if __name__ == "__main__":

    args = parse_args()
    logging_util.set_environ(args)
    
    names = ["GSE96583_1", "GSE96583_2", "GSE96583_3", "GSE96583_4", "GSE96583_5", "GSE130148"]

    for name in names:
        helpers.set_seed(args.seed)
        allset, label_dict = \
            MarkerGenePredictionDataset.create_dataset(name, add_marker_genes=True)
        model = helpers.load_model_inference(args)
        cell_label2marker = get_cell_type_label2marker(name)

        auroc_list = []
        for data in tqdm(allset):
            # try to perturb every gene in the cell, one gene each time
            pred_logits = get_perturbed_logits(args, model,
                                data["input_ids"], label_dict, cell_label2marker)
            # 1 - cosine similarity (btw/ original cell and its perturbed versions)
            pred_scores = 1 - torch.sum(pred_logits[0:1] * pred_logits[1:], dim=-1)
            
            marker_gene_label = np.isin(data["input_ids"][1:], data["marker_genes"])
            
            if marker_gene_label.sum() > 0:
                auroc = roc_auc_score(marker_gene_label, pred_scores.cpu().numpy(), average='macro') # insensitive to class imbalance
                auroc_list.append(auroc)
                
        auroc = sum(auroc_list) / len(auroc_list)
        print("auroc: ", round(auroc, 4))