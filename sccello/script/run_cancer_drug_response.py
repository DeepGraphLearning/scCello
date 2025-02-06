import os
import sys
import argparse
import numpy as np
import logging

from transformers import Trainer

EXC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(EXC_DIR)
from sccello.src.utils import config as utils_config
from sccello.src.utils import helpers, logging_util
from sccello.src.data.dataset import CancerDrugResponseDataset
from sccello.src.collator.collator_for_classification import DataCollatorForCellClassification

logging.basicConfig(level=logging.INFO)

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", default="cancer_drug_response",
                        choices=["cancer_drug_response"])
    parser.add_argument("--training_config", help="huggingface training configuration file",
                            default=os.path.join(EXC_DIR, f"./sccello/configs/cell_level/"
                                                 "cancer_drug_response_bert_nonparametric.json"))
    parser.add_argument("--pretrained_ckpt", type=str, help="pretrained model checkpoints", required=True)
    parser.add_argument("--output_dir", type=str, default="./single_cell_output", )
    parser.add_argument("--seed", help="random seed", type=int, default=0) # seed Follow DeepDCR

    parser.add_argument("--pass_cell_cls", type=int, default=1)
    parser.add_argument("--normalize", type=int, default=1)

    parser.add_argument("--data_source", type=str, help="specify which dataset is being used")
    parser.add_argument("--model_source", type=str, default="model_prototype_contrastive")

    parser.add_argument("--wandb_project", help="wandb project name", type=str, default="cancer_drug_response")
    parser.add_argument("--wandb_run_name", help="wandb run name", type=str, 
                        default="test", required=True)
    args = parser.parse_args()

    if args.pretrained_ckpt.endswith("/"):
        args.pretrained_ckpt = args.pretrained_ckpt[:-1]

    # model_source
    args.model_class = {
        "model_prototype_contrastive": "PrototypeContrastiveForMaskedLM"
    }[args.model_source]

    print("args: ", args)

    return args

if __name__ == "__main__":

    args = parse_args()
    logging_util.set_environ(args)
    helpers.set_seed(args.seed)
    
    name = "DeepCDR_data"
    feat, data_idx = CancerDrugResponseDataset.create_dataset(name)
    mutation_feature, drug_feature, gexpr_feature, methylation_feature = feat

    dataset = CancerDrugResponseDataset.from_csv(gexpr_feature)
    args.data_source = name

    # define output directory path
    args.output_dir = helpers.create_downstream_output_dir(args)

    training_args, training_cfg = utils_config.build_training_args(args, metric_for_best_model=f"{args.data_source}/macro_f1")

    model = helpers.load_model_inference(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForCellClassification(model_input_name=["input_ids"]),
    )
    print("args: ", args)

    # inference
    h_data = trainer.predict(dataset)
    h_data = h_data.predictions[1][:, -1, :]

    h_data, tag = helpers.cellrepr_post_process(model, h_data, pass_cell_cls=args.pass_cell_cls, normalize=args.normalize)
    
    np.save(f'{EXC_DIR}/embedding_storage/cdr_{tag}_embedding.npy', h_data)
    print(f'save to {EXC_DIR}/embedding_storage/cdr_{tag}_embedding.npy')

    # further move to DeepCDR's repo for evaluation