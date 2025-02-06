import os
import sys
import argparse
import psutil
import pickle

from datasets import load_from_disk

EXC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(EXC_DIR)

from sccello.src.data.data_proc import ProcessSingleCellData
from sccello.src.utils import helpers
from sccello.src.data.tokenizer import TranscriptomeTokenizer

TRUNCATE_LENGTH=2048

def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--h5ad_data_path", default="./data/example_h5ad/")
    parser.add_argument("--save_dir", default="./data/example_data_saved")
    parser.add_argument("--num_proc", default=12)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    """
    Transform single cell data from .h5ad format to huggingface dataset
    """

    args = parse_args()
    data_file = os.path.join(args.save_dir, "proceseed_pretraining_data")
    if not os.path.exists(data_file):
        pretrain_dataset = ProcessSingleCellData.from_h5ad_dir_caching(
                h5ad_data_dir=args.h5ad_data_path, save_dir=args.save_dir)
        pretrain_dataset.save_to_disk(data_file)
    else:
        pretrain_dataset = load_from_disk(data_file)

    print("======== Processed data ========")
    print(pretrain_dataset)
    print("cpu_mem: ", psutil.cpu_percent())
    print("used% ", psutil.virtual_memory().percent)
    print("avail% ", psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)

    def measure_length(example):
        example["length"] = [len(x) for x in example["gene_token_ids"]]
        return example
    pretrain_dataset = pretrain_dataset.map(measure_length, 
                            num_proc=args.num_proc, batched=True, batch_size=2048)

    proc_label_type = "cell_type"
    pretrain_dataset, label_type_idmap = helpers.process_label_type(pretrain_dataset, 12, proc_label_type)
    idmap_file = os.path.join(args.save_dir, f"./pretrain_{proc_label_type}_idmap.pkl")
    pickle.dump(label_type_idmap, open(idmap_file, "wb"))

    tk = TranscriptomeTokenizer(num_proc=args.num_proc, truncate_length=TRUNCATE_LENGTH)
    pretrain_dataset = tk.tokenize(pretrain_dataset)

    data_file = os.path.join(args.save_dir, "proceseed_pretraining_data_postproc")
    pretrain_dataset.save_to_disk(data_file)
