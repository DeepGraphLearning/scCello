
from functools import partial
from sccello.src import utils

class TranscriptomeTokenizer:

    def __init__(self, num_proc=4, truncate_length=2048):

        self.token_vocab_dict = utils.data_loading.get_prestored_data("token_dictionary_pkl")
        self.num_proc = num_proc
        self.truncate_length = truncate_length

    def tokenize(self, dataset):
        dataset = dataset.rename_column("gene_token_ids", "input_ids")
        
        # truncate dataset
        non_list_property = set([
            "length", "cell_counts", "cell_assay_ids",
            "cell_donor_local_ids", "cell_dataset_id", "cell_disease", 
            "cell_ct_ontology", "cell_type",
            "cell_tissue", "cell_tissue_ontology", "cell_dev",
        ])
        def truncate(example, truncate_length: int = None):
            key_list = [k for k in dataset.features.keys() if k not in non_list_property]
            
            for k in key_list:
                example[k] = example[k][:truncate_length]
            return example

        dataset = dataset.map(
            partial(truncate, truncate_length=self.truncate_length), 
            num_proc=self.num_proc
        )
        
        # measure lengths of dataset
        def measure_length(example, truncate_length: int=None):
            if "length" not in example:
                example["length"] = len(example["input_ids"])
            example["length"] = min(example["length"], truncate_length)
            return example

        dataset = dataset.map(
            partial(measure_length, truncate_length=self.truncate_length), 
            num_proc=self.num_proc, batched=False,
        )

        return dataset
