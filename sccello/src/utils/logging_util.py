import os
import psutil
import logging

import datasets

import wandb

EXC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def init_wandb(args):
    wandb.init(project=args.wandb_project, dir=args.wandb_dir, name=args.wandb_run_name)

def set_environ(args):

    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["OMPI_MCA_opal_cuda_support"] = "true"
    os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"

    # create wandb related arguments
    args.wandb_project = args.objective
    args.wandb_run_name = "/".join([args.objective, args.wandb_run_name])
    args.wandb_dir = os.path.join(args.output_dir, f"wandb/{args.objective}")
    args.wandb_cache_dir = os.path.join(args.output_dir, f"wandb_cache/{args.objective}")
    args.wandb_config_dir = os.path.join(args.output_dir, f"wandb_config/{args.objective}")
    if not os.path.exists(args.wandb_dir):
        os.makedirs(args.wandb_dir, exist_ok=True)
    if not os.path.exists(args.wandb_cache_dir):
        os.makedirs(args.wandb_cache_dir, exist_ok=True)
    if not os.path.exists(args.wandb_config_dir):
        os.makedirs(args.wandb_config_dir, exist_ok=True)

    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_DIR"] = args.wandb_dir
    os.environ["WANDB_CACHE_DIR"] = args.wandb_cache_dir
    os.environ["WANDB_CONFIG_DIR"] = args.wandb_config_dir

    os.environ["WANDB__SERVICE_WAIT"] = "300"

def basic_info_logging(model):
    logging.info(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    logging.info(f"#params: {sum([p.numel() for p in model.parameters()])}")

def pretraining_start_info_logging(train_test_dataset, model):

    logging.info("Starting training.")

    basic_info_logging(model)

    files = train_test_dataset["train"].cache_files
    logging.info(f"Number of cache files in dataset : {len(files)}")
    size_gb = sum([os.path.getsize(x["filename"]) / (1024 ** 3) for x in files])
    logging.info(f"Dataset size (cache file) : {size_gb:.2f} GB")
    
    # Log Hugging Face datasets memory limit
    logging.info(f"IN_MEMORY_MAX_SIZE: {datasets.config.IN_MEMORY_MAX_SIZE}")
    
    logging.info(f"#training data samples: {len(train_test_dataset['train'])}")
    logging.info(f"#test data samples: {len(train_test_dataset['test'])}")

    
