import os
import sys
EXC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(EXC_DIR)

from sccello.src.model_prototype_contrastive import PrototypeContrastiveForMaskedLM

model = PrototypeContrastiveForMaskedLM.from_pretrained("katarinayuan/scCello-zeroshot", output_hidden_states=True)
