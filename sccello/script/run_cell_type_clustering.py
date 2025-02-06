import os
import sys
import argparse
import pickle
import logging
import subprocess
import ipdb

import torch
from transformers import Trainer

EXC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(EXC_DIR)

from sccello.src.utils import config as utils_config
from sccello.src.utils import helpers, logging_util, evaluating, data_loading
from sccello.src.data.dataset import CellTypeClassificationDataset
from sccello.src.collator.collator_for_classification import DataCollatorForCellClassification

logging.basicConfig(level=logging.INFO)

def parse_args():

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--objective", type=str, default="cell_type_clustering", 
                            choices=["cell_type_clustering"])
    parser.add_argument("--training_config", help="huggingface training configuration file",
                            default=os.path.join(EXC_DIR, f"./sccello/configs/cell_level/"
                                                 "cell_type_classification_bert_clustering.json"))
    parser.add_argument("--pretrained_ckpt", type=str, help="pretrained model checkpoints", required=True)
    parser.add_argument("--output_dir", type=str, default="./single_cell_output", required=True)
    parser.add_argument("--seed", help="random seed", type=int, default=42)

    parser.add_argument("--data_branch", type=str, default="frac100", choices=["frac100"])
    parser.add_argument("--data_source", type=str, help="specify which dataset is being used")
    parser.add_argument("--model_source", type=str, default="model_prototype_contrastive")
    
    parser.add_argument("--indist", type=int, default=0)
    parser.add_argument("--normalize", type=int, default=0)
    parser.add_argument("--pass_cell_cls", type=int, default=0)

    parser.add_argument("--wandb_project", help="wandb project name", type=str, default="cell_type_clustering")
    parser.add_argument("--wandb_run_name", help="wandb run name", type=str, 
                        default="test", required=True)
    args = parser.parse_args()

    if args.pretrained_ckpt.endswith("/"):
        args.pretrained_ckpt = args.pretrained_ckpt[:-1]

    args.model_class = {
        "model_prototype_contrastive": "PrototypeContrastiveForMaskedLM",
    }[args.model_source]

    print("args: ", args)

    # create directory to store embeddings
    subprocess.call(f"mkdir -p {os.path.join(EXC_DIR, './embedding_storage')}", shell=True)

    return args

def solve_clustering(args, all_datasets):
    trainset, test_data1, test_data2, label_dict = all_datasets

    args.output_dir = helpers.create_downstream_output_dir(args)

    training_args, training_cfg = utils_config.build_training_args(args, metric_for_best_model=f"{args.data_source}/macro_f1")
    model = helpers.load_model_inference(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForCellClassification(model_input_name=["input_ids"], pad_to_multiple_of=None)
    )

    all_result_dict = {}
    test_data1 = test_data1.rename_column("label", "label_cell_type")
    test_data2 = test_data2.rename_column("label", "label_cell_type")
    for setting, dataset_to_test in zip(["data1", "data2"], [test_data1, test_data2]):
        if args.indist:
            file = f"./embedding_storage/cellreprs_indist_{args.data_source}_{setting}_try2.pkl"
        else:
            file = f"./embedding_storage/cellreprs_{args.data_source}_{setting}_try2.pkl"

        ipdb.set_trace()
        if os.path.exists(file):
            h_embed = pickle.load(open(file, "rb"))
        else:
            h_embed = trainer.predict(dataset_to_test)
            h_embed = h_embed.predictions[1][:, -1, :]
            pickle.dump(h_embed, open(file, "wb"))
        
        h_embed, tag = helpers.cellrepr_post_process(model, h_embed, pass_cell_cls=args.pass_cell_cls, normalize=args.normalize)
        ipdb.set_trace()

        # inferenced embedding is further used in novel cell type classification, 
        # but is not for cell type clustering evaluation
        if args.indist and setting == "data1":
            continue

        batch_effect_results_dict = evaluating.eval_scanpy_gpu_metrics((h_embed, dataset_to_test["cell_dataset_id"]), 
                                        fast_mode=True, use_anndata=False, verbose=1, batch_effect=True)
        batch_effect_results_dict = {k.replace("label", "batch"):v for k,v in batch_effect_results_dict.items()}
        clustering_results_dict = evaluating.eval_scanpy_gpu_metrics((h_embed, dataset_to_test["label_cell_type"]), 
                                        fast_mode=True, use_anndata=False, verbose=1)
        print(f"[fast test]: {setting}_results_dict: ", batch_effect_results_dict)
        print(f"[fast test]: {setting}_results_dict: ", clustering_results_dict)
        all_result_dict = {**all_result_dict, **batch_effect_results_dict, **clustering_results_dict}

    print("all_results: ", all_result_dict)


if __name__ == "__main__":

    args = parse_args()
    logging_util.set_environ(args)
    
    assert args.data_branch.startswith("frac")

    names = CellTypeClassificationDataset.subsets["frac"]
    if args.indist:
        names = [names[0]]
        
    for name in names:
        # every data is tested under the same seeded setting
        helpers.set_seed(args.seed)
        args.data_source = f"frac_{name}"
        all_datasets = data_loading.get_fracdata(name, args.data_branch, args.indist, False)
        solve_clustering(args, all_datasets)