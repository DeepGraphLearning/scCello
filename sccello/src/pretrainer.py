"""
Huggingface data collator and trainer modified to accommodate single-cell transcriptomics data.
"""
from typing import Optional

import numpy as np
import torch
import datasets
from packaging import version
from torch.utils.data.sampler import RandomSampler
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
)
from transformers.file_utils import is_datasets_available, is_sagemaker_dp_enabled
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import has_length

if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as dist
else:
    import torch.distributed as dist

_is_torch_generator_available = False
if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True

from sccello.src.collator.collator_for_pretraining import scCelloPreCollator

class scCelloPretrainer(Trainer):
    def __init__(self, *args, **kwargs):
        data_collator = kwargs.get("data_collator",None)
        self.model_input_name = kwargs.pop("model_input_name")

        if data_collator is None:
            precollator = scCelloPreCollator(model_input_names=self.model_input_name)
            mlm_probability = kwargs.pop("mlm_probability")
            pad_to_multiple_of = kwargs.pop("pad_to_multiple_of")
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=precollator, mlm=True, mlm_probability=mlm_probability,
                pad_to_multiple_of=pad_to_multiple_of
            )
            kwargs["data_collator"] = data_collator

        super().__init__(*args, **kwargs)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        else:
            return RandomSampler(self.train_dataset)
