import os
import ast
import json
import jinja2
import easydict
import yaml
from jinja2 import meta
import argparse
import json
import shutil

from transformers import TrainingArguments

def load_json(cfg_file):
    with open(cfg_file, "r") as fin:
        configs = json.load(fin)
    return configs

def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw_text = fin.read()
    env = jinja2.Environment()
    ast_ = env.parse(raw_text)
    vars = meta.find_undeclared_variables(ast_)
    return vars

def literal_eval(string):
    """Evaluate an expression into a Python literal structure."""
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string

def deal_dynamic_args(parser):
    """Capable of modifying the arguments in ds_config as well."""

    args, unparsed = parser.parse_known_args()
    
    # get dynamic arguments defined in the ds_config file
    vars = detect_variables(args.ds_config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars, still_unparsed = parser.parse_known_args(unparsed)
    vars = {k: literal_eval(v) for k, v in vars._get_kwargs()}

    assert len(still_unparsed) == 0, f"[Warning]: wrong argument names: {still_unparsed}"

    return args, vars

def complete_ds_config(cfg_file, output_dir, context=None):
    with open(cfg_file, "r") as fin:
        raw_text = fin.read()
        
    if context is not None:
        template = jinja2.Template(raw_text)
        instance = template.render(context)
        configs = easydict.EasyDict(yaml.safe_load(instance))
    else:
        configs = easydict.EasyDict(yaml.safe_load(raw_text))

    old_cfg_file = os.path.join(output_dir, "ds_config_template.json")
    shutil.copy(cfg_file, old_cfg_file)
    
    new_cfg_file = os.path.join(output_dir, "ds_config.json")
    with open(new_cfg_file, "w") as fout:
        json.dump(configs, fout)

    return new_cfg_file, configs

def build_training_args(args, metric_for_best_model=None):

    training_cfg = load_json(args.training_config)
    
    training_cfg["run_name"] = args.wandb_run_name
    if hasattr(args, "deepspeed") and args.deepspeed:
        training_cfg["deepspeed"] = args.ds_config_completed
    training_cfg["output_dir"] = args.output_dir
    training_cfg["ddp_timeout"] = 180000
    if metric_for_best_model is not None:
        training_cfg["metric_for_best_model"] = metric_for_best_model

    for key in training_cfg.keys():
        match_key = "change_" + key
        if hasattr(args, match_key) and getattr(args, match_key) is not None:
            training_cfg[key] = getattr(args, match_key)
    
    if training_cfg["num_train_epochs"] == 0:
        training_cfg["do_train"] = False

    # ensure not overwriting previously saved model
    model_saving_file = os.path.join(args.output_dir, "pytorch_model.bin")
    if os.path.isfile(model_saving_file) is True:
        raise Exception("Model already saved to this directory.")

    training_args = TrainingArguments(**training_cfg)

    return training_args, training_cfg