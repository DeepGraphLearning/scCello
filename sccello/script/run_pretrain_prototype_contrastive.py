import os
import sys
import argparse
import numpy as np
import logging
import ipdb
import wandb

import torch
from datasets import load_from_disk, load_dataset
from transformers import BertConfig

EXC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(EXC_DIR)

from sccello.src.utils import config as utils_config
from sccello.src.utils import helpers, logging_util, data_loading
from sccello.src.pretrainer import scCelloPretrainer
from sccello.src.model_prototype_contrastive import PrototypeContrastiveForMaskedLM

logging.basicConfig(level=logging.INFO)

def parse_args():

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--objective", type=str, 
                        default="plm", choices=["plm"])
    parser.add_argument("--ds_config", help="deepspeed configuration file", 
                        default="./ds_config_zero.json")
    parser.add_argument("--model_config", help="huggingface model configuration file", 
                        default=os.path.join(EXC_DIR, "./sccello/configs/pretrain/sccello_bert_model.json"))
    parser.add_argument("--training_config", help="huggingface training configuration file",
                        default=os.path.join(EXC_DIR, "./sccello/configs/pretrain/sccello_bert_training.json"))
    parser.add_argument("--output_dir", type=str, default="./single_cell_output", required=True)
    parser.add_argument("--seed", help="random seed", type=int, default=237)

    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--tau", type=float, default=0.01)
    
    parser.add_argument("--change_learning_rate", type=float, default=None)
    parser.add_argument("--change_per_device_train_batch_size", type=int, default=12)
    parser.add_argument("--change_gradient_accumulation_steps", type=int, default=None)

    parser.add_argument("--data_branch", type=str, default="frac100", choices=["frac100"])

    parser.add_argument("--wandb_project", help="wandb project name", type=str, default="single_cell")
    parser.add_argument("--wandb_run_name", help="wandb run name", type=str, required=True)

    args, vars = utils_config.deal_dynamic_args(parser)
    logging.info(args)

    return args, vars

def build_model_cfg(args):

    model_cfg = utils_config.load_json(args.model_config)
    
    token_dictionary = data_loading.get_prestored_data("token_dictionary_pkl")
    model_cfg["pad_token_id"] = token_dictionary.get("<pad>")

    training_cfg = utils_config.load_json(args.training_config)
    model_cfg["total_logging_steps"] = training_cfg["logging_steps"] * training_cfg["gradient_accumulation_steps"]
    model_cfg["bert_objective"] = args.objective
    model_cfg["tau"] = args.tau
    model_cfg["data_branch"] = args.data_branch

    return model_cfg

def load_pretrain_dataset(args):

    train_dataset = load_dataset("katarinayuan/scCello_pretrain_unsplitted")["train"]
    train_dataset = train_dataset.rename_column("gene_token_ids", "input_ids")
    # avoid unnecessary data transfer
    train_dataset = train_dataset.remove_columns(
        ["gene_expression_nums", "cell_dataset_id", "cell_disease", 
        "cell_assay_ids", "cell_donor_local_ids", "cell_ct_ontology", "cell_tissue", 
        "cell_tissue_ontology", "cell_dev", "cell_counts"]
    )
    return train_dataset

def build_trainer_pretrain(
    args, training_args, training_cfg, train_test_dataset, model,
):
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(labels, np.ndarray) or isinstance(labels, torch.Tensor):
            labels = [labels]
        metrics = {}
        for label_name, pred, label in zip(training_cfg["label_names"], preds, labels):
            mask = label != -100
            acc = (pred == label)[mask].mean()
            metric_name = label_name.split("labels")[0] + "acc"
            metrics[metric_name] = acc

        return metrics
    
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, np.ndarray) or isinstance(logits, torch.Tensor):
            logits = [logits]
        pred_ids = []
        for logit in logits:
            pred_id = torch.argmax(logit, dim=-1)
            pred_ids.append(pred_id)
        return pred_ids

    trainer = scCelloPretrainer(
        model=model,
        args=training_args,
        train_dataset=train_test_dataset["train"],
        eval_dataset=train_test_dataset["test"],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        model_input_name=["input_ids", "cell_type"],
        mlm_probability=args.mlm_probability,
        pad_to_multiple_of=None,
    )

    return trainer


if __name__ == "__main__":

    args, vars = parse_args()
    logging_util.set_environ(args)
    helpers.set_seed(args.seed)

    # logging_util.init_wandb(args)
    
    # build model config
    model_cfg = build_model_cfg(args)

    # build model
    bert_cfg = BertConfig(**model_cfg)
    model = PrototypeContrastiveForMaskedLM(bert_cfg).train()
    
    # assign output directory
    args.output_dir = helpers.create_pretraining_output_dir(args, model_cfg)

    # re-write the ds_config file with passed args
    args.ds_config_completed, ds_cfg = utils_config.complete_ds_config(args.ds_config, args.output_dir, context=vars)

    # build training data and validation data
    train_dataset = load_pretrain_dataset(args)
    train_test_dataset = train_dataset.train_test_split(test_size=0.001, seed=237) # as used in scCello

    # build trainer
    training_args, training_cfg = utils_config.build_training_args(args)
    trainer = build_trainer_pretrain(args, training_args, training_cfg, train_test_dataset, model)

    # training
    logging_util.pretraining_start_info_logging(train_test_dataset, model)
    trainer.train()
    trainer.save_model(training_cfg["output_dir"])

    print("sync dir: ", wandb.__dict__['run'].__dict__['_settings']['sync_dir'])