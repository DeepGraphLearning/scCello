import os
import pytz
import datetime
import subprocess
import random
import numpy as np

import torch
import torch.nn.functional as F

EXC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

from sccello.src.utils import logging_util


def set_seed(seed):
    """Sets random seed for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model_setting_card(args, model_cfg):

    num_layers = model_cfg["num_hidden_layers"]
    num_heads = model_cfg["num_attention_heads"]
    hidden_dim = model_cfg["hidden_size"]
    trunc_length = model_cfg["max_position_embeddings"]
    attn_dropout = model_cfg["attention_probs_dropout_prob"]
    hidden_dropout = model_cfg["hidden_dropout_prob"]

    model_setting = f"layer{num_layers}_heads{num_heads}" \
                    f"_dim{hidden_dim}_length{trunc_length}_attndrop{attn_dropout}" \
                    f"_hiddrop{hidden_dropout}"
    
    return model_setting

def create_pretraining_output_dir(args, model_cfg):

    timezone = pytz.timezone("US/Eastern")
    cur_date = datetime.datetime.now(tz=timezone)
    datestamp = f"{str(cur_date.year)[-2:]}{cur_date.month:02d}{cur_date.day:02d}_{cur_date.strftime('%X').replace(':','')}"

    model_setting_card = create_model_setting_card(args, model_cfg)
    output_dir = os.path.join(args.output_dir, model_setting_card, datestamp)
    subprocess.call(f"mkdir -p {output_dir}", shell=True)

    return output_dir


def create_downstream_output_dir(args):

    timezone = pytz.timezone("US/Eastern")
    cur_date = datetime.datetime.now(tz=timezone)
    datestamp = f"{str(cur_date.year)[-2:]}{cur_date.month:02d}{cur_date.day:02d}_{cur_date.strftime('%X').replace(':','')}"

    ckpt_card = args.pretrained_ckpt.split("single_cell_output/")[-1]

    output_dir = os.path.join(
        args.output_dir,
        args.objective,
        args.data_source,
        ckpt_card, datestamp
    )

    print(f"\n\n\noutput_dir: {output_dir}\n\n\n")

    subprocess.call(f"mkdir -p {output_dir}", shell=True)

    return output_dir


def load_model_inference(args):
    
    model = eval(args.model_class).from_pretrained(args.pretrained_ckpt, output_hidden_states=True).to("cuda")
    for param in model.bert.parameters():
        param.requires_grad = False
    logging_util.basic_info_logging(model)

    return model

def cellrepr_post_process(model, h_data, pass_cell_cls=False, normalize=False):

    assert isinstance(h_data, np.ndarray)
    h_data = torch.tensor(h_data, device=model.device)
    tag = ""
    if pass_cell_cls:
        batch_size = 512
        module = model.cell_cls
        new_h_data = []
        for i in range(0, len(h_data), batch_size):
            new_h_data.append(module(h_data[i: i+batch_size]))
        h_data = torch.cat(new_h_data, dim=0).detach()
        tag += "_passcellcls"
    if normalize:
        h_data = F.normalize(h_data, p=2, dim=-1)
        tag += "_normalized"
    h_data = h_data.cpu().numpy()

    return h_data, tag

def labels_category2int(labels, return_map=False):
    labels = [str(_) for _ in labels]
    label_type_list = list(set(labels))
    label_type_list = sorted(label_type_list)
    label_type_idmap = {label_type_list[i]: i for i in range(len(label_type_list))}     
    labels_num = np.array([label_type_idmap[x] for x in labels])
    if return_map:
        return labels_num, label_type_idmap
    return labels_num

def process_label_type(train_dataset, num_proc=None, proc_label_type="cell_type"):

    if proc_label_type == "tech_sample":
        def add_tech_sample(example):
            example["tech_sample"] = example["cell_dataset_id"] + "_" + str(example["cell_donor_local_ids"])
            return example
        train_dataset = train_dataset.map(add_tech_sample, num_proc=8, batched=False, desc="tech_sample")

    label_type_list = list(set(train_dataset[proc_label_type]))
    label_type_idmap = {label_type_list[i]: i for i in range(len(label_type_list))}

    def collect_cell(example):
        example[proc_label_type] = label_type_idmap[example[proc_label_type]]
        return example
    train_dataset = train_dataset.map(collect_cell, num_proc=num_proc, desc="mapping")

    return train_dataset, label_type_idmap