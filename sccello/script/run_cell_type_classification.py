import os
import sys
import argparse
import numpy as np
import logging
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score
)

import torch
from transformers import Trainer

EXC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(EXC_DIR)

from sccello.src.utils import config as utils_config
from sccello.src.utils import helpers, logging_util, data_loading
from sccello.src.model_prototype_contrastive import PrototypeContrastiveForSequenceClassification
from sccello.src.collator.collator_for_classification import DataCollatorForCellClassification

logging.basicConfig(level=logging.INFO)

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", default="cell_type_classification", 
                        choices=["cell_type_classification"])
    parser.add_argument("--training_type", default="linear_probing", 
                            choices=["linear_probing", "from_scratch_linear"])
    parser.add_argument("--training_config", help="huggingface training configuration file")
    parser.add_argument("--pretrained_ckpt", type=str, help="pretrained model checkpoints", required=True)
    parser.add_argument("--output_dir", type=str, default="./single_cell_output", required=True)
    parser.add_argument("--seed", help="random seed", type=int, default=42)

    parser.add_argument("--data_branch", default="frac100", choices=["frac100"])
    parser.add_argument("--data_source", type=str, help="specify which dataset is being used")
    parser.add_argument("--model_source", type=str, default="model_prototype_contrastive")

    parser.add_argument("--indist", type=int, default=1)
    parser.add_argument("--normalize", type=int, default=0)
    parser.add_argument("--pass_cell_cls", type=int, default=0)
    parser.add_argument("--batch_effect", type=int, default=None)
    parser.add_argument("--further_downsample", type=float, default=0.01)
    
    ### change configurations in the training config yaml file ###
    parser.add_argument("--change_num_train_epochs", type=int, default=None)
    parser.add_argument("--change_learning_rate", type=float, default=None)
    parser.add_argument("--change_per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--change_lr_scheduler_type", type=str, default=None)

    parser.add_argument("--wandb_project", help="wandb project name", type=str, default="cell_type_classification")
    parser.add_argument("--wandb_run_name", help="wandb run name", type=str, 
                        default="test", required=True)
    args = parser.parse_args()

    # training_config
    file_name = "./sccello/configs/cell_level/cell_type_classification_bert_training"
    if args.training_type == "linear_probing":
        file_name += "_probing"
    args.training_config = os.path.join(EXC_DIR, f"{file_name}.json")

    if args.pretrained_ckpt.endswith("/"):
        args.pretrained_ckpt = args.pretrained_ckpt[:-1]

    # model_source
    args.model_class = {
        "model_prototype_contrastive": "PrototypeContrastiveForSequenceClassification",
    }[args.model_source]

    print("args: ", args)

    return args


def build_supervised_trainer(args, training_args, model, train_dataset, eval_dataset):

    def compute_metrics(eval_preds):
        probs, labels = eval_preds
        preds = np.argmax(probs, axis=-1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average='macro')
        try:
            if probs.shape[1] > 2:
                auroc = roc_auc_score(labels, probs, multi_class="ovo", average='macro') # insensitive to class imbalance
            else:
                auroc = roc_auc_score(labels, probs[:, 1], average='macro') # insensitive to class imbalance
        except:
            auroc = 0

        return {
            f'{args.data_source}/accuracy': acc,
            f'{args.data_source}/macro_f1': macro_f1,
            f'{args.data_source}/auroc': auroc
        }

    def preprocess_logits_for_metrics(logits, labels):
        probs = torch.softmax(logits, dim=-1)
        return probs

    # create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForCellClassification(model_input_name=["input_ids"], pad_to_multiple_of=None),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    return trainer


def load_model_supervised(args, training_args, training_cfg, label_dict):

    model_kwargs = {
        "num_labels": len(label_dict.keys()), # variable in huggingface
        "total_logging_steps": training_cfg["logging_steps"] * training_args.gradient_accumulation_steps,
        "data_source": args.data_source,
        "normalize_flag": args.normalize,
        "pass_cell_cls": args.pass_cell_cls,
    }

    if args.training_type in ["from_scratch_linear"]:
        cfg, model_kwargs = eval(args.model_class).config_class.from_pretrained(
                                args.pretrained_ckpt, return_unused_kwargs=True, **model_kwargs)
        model = eval(args.model_class)(cfg, **model_kwargs).to("cuda")
    elif args.training_type in ["linear_probing"]:
        model = eval(args.model_class).from_pretrained(args.pretrained_ckpt, **model_kwargs).to("cuda")
        for param in model.bert.parameters():
            param.requires_grad = False
    else:
        raise NotImplementedError

    logging_util.basic_info_logging(model)

    return model


def solve_classification_supervised(args, all_datasets):

    trainset, evalset, testset, label_dict = all_datasets
    trainset = trainset.train_test_split(train_size=0.1, seed=args.seed)["train"]
    
    # further downsample 
    trainset = trainset.train_test_split(train_size=args.further_downsample, seed=args.seed)["train"]
    evalset = evalset.train_test_split(train_size=args.further_downsample, seed=args.seed)["train"]

    # define output directory path
    args.output_dir = helpers.create_downstream_output_dir(args)
    
    training_args, training_cfg = utils_config.build_training_args(args, metric_for_best_model=f"{args.data_source}/macro_f1")
    assert training_cfg["do_eval"] is True and training_cfg["do_train"] is True

    model = load_model_supervised(args, training_args, training_cfg, label_dict)

    trainer = build_supervised_trainer(args, training_args, model, trainset, evalset)
    trainer.train()

    trainer.evaluate(evalset) # metric_key_prefix default as "eval"
    eval_predictions = trainer.predict(evalset)
    trainer.evaluate(testset, metric_key_prefix="test")
    test_predictions = trainer.predict(testset)
    print("eval_predictions.metrics: ", eval_predictions.metrics)
    print("test_predictions.metrics: ", test_predictions.metrics)
    
    return {**eval_predictions.metrics, **test_predictions.metrics}


if __name__ == "__main__":

    args = parse_args()
    logging_util.set_environ(args)

    logging_util.init_wandb(args)
    
    assert args.data_branch.startswith("frac")

    helpers.set_seed(args.seed)
    args.data_source = f"frac_indist"
    all_datasets = data_loading.get_fracdata("celltype", args.data_branch, args.indist, args.batch_effect)
    
    solve_classification_supervised(args, all_datasets)
    
    # print("sync dir: ", wandb.__dict__['run'].__dict__['_settings']['sync_dir'])