from typing import Optional, Tuple, Union
import wandb
import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers.models.bert.modeling_bert import (
    BertModel, BertEncoder, 
    BertLayer, BertIntermediate, BertOutput, BertSelfOutput,
    BertSelfAttention, BertAttention,
    BertPreTrainedModel
)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    MaskedLMOutput,
    SequenceClassifierOutput
)
from transformers.utils import logging

from torchmetrics.functional.classification import (
    multiclass_accuracy, multiclass_auroc, f1_score
)

from sccello.src import cell_ontology

logger = logging.get_logger(__name__)


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        return first_token_tensor

class PrototypeContrastiveEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)        
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # legacy
        self.tf_class_embeddings = nn.Embedding(1, config.hidden_size, scale_grad_by_freq=True)
        self.tf_superclass_embeddings = nn.Embedding(1, config.hidden_size, scale_grad_by_freq=True)
        self.expbin_embeddings = nn.Embedding(1, config.hidden_size, scale_grad_by_freq=True, padding_idx=config.pad_token_id)
    
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # legacy
        tf_class_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        tf_superclass_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        expbin_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        
        tf_class_embeddings = self.tf_class_embeddings(tf_class_ids)
        tf_superclass_embeddings = self.tf_superclass_embeddings(tf_superclass_ids)
        expbin_embeddings = self.expbin_embeddings(expbin_ids)

        embeddings = inputs_embeds + tf_class_embeddings + tf_superclass_embeddings + expbin_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class PrototypeContrastiveSelfAttention(BertSelfAttention):

    def __init__(self, config, position_embedding_type=None):
        super(PrototypeContrastiveSelfAttention, self).__init__(config, position_embedding_type=position_embedding_type)

class PrototypeContrastiveAttention(BertAttention):
    def __init__(self, config, position_embedding_type=None):
        nn.Module.__init__(self)
        self.self = PrototypeContrastiveSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

class PrototypeContrastiveLayer(BertLayer):

    def __init__(self, config):
        nn.Module.__init__(self)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PrototypeContrastiveAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = PrototypeContrastiveAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

class PrototypeContrastiveEncoder(BertEncoder):

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.layer = nn.ModuleList([PrototypeContrastiveLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

class PrototypeContrastiveModel(BertModel):

    def __init__(self, config, add_pooling_layer=True):
        BertPreTrainedModel.__init__(self, config)
        self.config = config

        self.embeddings = PrototypeContrastiveEmbeddings(config)
        self.encoder = PrototypeContrastiveEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        # Initialize weights and apply final processing
        self.post_init()

class PrototypeContrastivePredictionHead(nn.Module):

    def __init__(self, config):
        super(PrototypeContrastivePredictionHead, self).__init__()
        self.dense_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        assert isinstance(config.hidden_act, str)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.dense_2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.hidden_size))
        self.dense_2.bias = self.bias
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)

        return hidden_states

class PrototypeContrastiveOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = PrototypeContrastivePredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class PrototypeContrastiveForMaskedLM(BertPreTrainedModel):

    """BERT-like encoder-based foundation model with cell-level contrastive
        learning and gene-level masked language modeling.
    """

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias", r"cls.predictions.decoder.weight"]
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = PrototypeContrastiveModel(config, add_pooling_layer=False)
        self.cls = PrototypeContrastiveOnlyMLMHead(config)
        self.cell_cls = PrototypeContrastiveOnlyMLMHead(config)

        # all cell types smaller than 600
        self.learned_centroid = nn.Embedding(600, config.hidden_size)
        self.affine_projection = nn.Linear(config.hidden_size, config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

        self.tau = config.tau
        self.data_branch = config.data_branch
        self.total_logging_steps = config.total_logging_steps
        self.step_count = 0

        # cell ontology graph related buffers
        anc_mask = cell_ontology.get_forbidden_cell_type_contrast(self.data_branch)
        self.register_buffer("cell_type_ancestor_mask", 
                             torch.tensor(anc_mask, device=self.device).detach())
        ctgraph_sim = cell_ontology.get_hierarchy_cell_type_contrast(self.data_branch)[0]
        self.register_buffer("ctgraph_sim", 
                             torch.tensor(ctgraph_sim, device=self.device).detach())

        # legacy
        self.binary_ce_loss_transform_cell = nn.Linear(1, 1)
        self.binary_ce_loss_transform_prototype = nn.Linear(1, 1)
        self.binary_ce_loss_transform_cellprot = nn.Linear(1, 1)
        

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
    
    def tie_weights(self):
        for module in self.modules():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()

    def _get_ground_truth_word_reprs(self):
        # in NLP, the word embedding (i.e., gene token embedding) weight is tied to the prediction layer weight
        z_truth = self.bert.embeddings.word_embeddings.weight # [word, dim]
        h_truth = self.cls(z_truth.unsqueeze(0))[0]
        return h_truth

    def _get_cell_representation(self, sequence_output):
        """Extract and normalize cell representations from sequence output."""
        # extract <cls> token representation
        h_cls = self.cell_cls(sequence_output)[:, 0, :] # [bsz, dim]
        return F.normalize(h_cls, p=2, dim=-1) / torch.sqrt(torch.tensor(self.tau, device=self.device)) 
    
    def _get_cell_type_representation(self, cell_type):
        """Retrieve and normalize cell type representations (prototypes)."""
        h_prot = self.learned_centroid(cell_type)
        return F.normalize(h_prot, p=2, dim=-1) / torch.sqrt(torch.tensor(self.tau, device=self.device))

    def _get_cell_representation_noised(self, bert_kwargs):
        """Pass input through encoder with dropout noise to help contrastive learning."""
        outputs_self = self.bert(**bert_kwargs)
        sequence_output_self = outputs_self[0]

        h_cls_self = self.cell_cls(sequence_output_self)[:, 0, :] # [bsz, dim]
        return F.normalize(h_cls_self, p=2, dim=-1) / torch.sqrt(torch.tensor(self.tau, device=self.device))

    def calculate_masked_gene_prediction(self, sequence_output, labels):
        """
        Compute masked token prediction using supervised contrastive learning.
        Args:
            sequence_output: [bsz, L, dim]
            labels: [bsz, L]
        """

        is_masked = labels != -100 if labels is not None else None # [bsz, L]

        # compute logits for masked positions
        h_mask = self.cls(sequence_output) # [bsz, L, dim]
        h_truth = self._get_ground_truth_word_reprs() # [vocab_size, dim]
        
        # normalize representations
        h_mask = F.normalize(h_mask, p=2, dim=-1) / torch.sqrt(torch.tensor(self.tau, device=self.device))
        h_truth = F.normalize(h_truth, p=2, dim=-1) / torch.sqrt(torch.tensor(self.tau, device=self.device))

        # compute supervised contrastive loss
        loss_fct = nn.CrossEntropyLoss()
        h_mask_flatten = h_mask.view(-1, self.config.hidden_size) # [bsz * L, dim]
        pred_scores_flatten = torch.matmul(h_mask_flatten, h_truth.t()) # [bsz * L, vocab_size]
        label_flatten = labels.view(-1) # [bsz * L]
        plm_loss = loss_fct(pred_scores_flatten, label_flatten)

        prediction_scores = pred_scores_flatten.view(h_mask.shape[0], h_mask.shape[1], -1) # [bsz, L, vocab_size]

        is_masked_flatten = is_masked.view(-1)
        masked_acc_plm = (pred_scores_flatten.argmax(dim=-1) == label_flatten)[is_masked_flatten].float().mean()

        return {
            "loss": plm_loss,
            "acc": masked_acc_plm,
            "prediction_scores": prediction_scores
        }
    
    def _ranking_contrastive_loss(self, cell_type, cls_self_contrast_scores):
        """Compute contrastive loss based on cell type relationships in ontology."""
    
        batch_size = cell_type.shape[0]
        cls_contrast_loss = []
        
        for i in range(batch_size):
            ctgraph_sim_level = self.ctgraph_sim[cell_type[i]][cell_type] # [bsz,]
            sim_level_list = sorted(list(set(ctgraph_sim_level.tolist())))[1:] # not include the minimum level
            
            for sim_level_idx in sim_level_list:
                # positive sample
                pos_indices = (ctgraph_sim_level >= sim_level_idx).nonzero().squeeze(1)
                cls_scores_pos = cls_self_contrast_scores[i][pos_indices]
                
                # negative sample
                neg_indices = ctgraph_sim_level < sim_level_idx
                neg_indices = torch.logical_and(neg_indices, 
                                self.cell_type_ancestor_mask[cell_type[i]][cell_type] == 0) # filter out ancestor on cell ontology graph
                neg_indices = neg_indices.nonzero().squeeze(1)
                cls_scores_neg = cls_self_contrast_scores[i][neg_indices]

                # concatenate logits and binary labels
                cls_scores = torch.cat([cls_scores_pos.unsqueeze(1), 
                                        cls_scores_neg.expand(len(cls_scores_pos), -1)], dim=-1) # [num_pos, num_neg]
                cls_contrast_labels = torch.zeros_like(cls_scores[:, 0], dtype=torch.long)
                
                cls_loss = torch.nn.CrossEntropyLoss(reduce=False)(cls_scores, cls_contrast_labels)
                cls_contrast_loss += cls_loss.unsqueeze(1)

        return torch.cat(cls_contrast_loss, dim=0).mean() if len(cls_contrast_loss) > 0 else torch.tensor(0.0, device=self.device)

    def calculate_relational_alignment(self, h_cls, cell_type, bert_kwargs):
        """
        Compute relational alignment loss using ranking-based contrastive learning.
        Args:
            h_cls: [bsz, dim]
            cell_type: [bsz,]
            bert_kwargs: dict
        """
        # get contrastive score matrix
        batch_size = len(h_cls)
        cls_self_index = torch.arange(batch_size, device=self.device) + dist.get_rank() * batch_size
        cls_self_mask = (cls_self_index.unsqueeze(1) == cls_self_index.unsqueeze(0)).int()
        assert (cls_self_mask == torch.eye(len(h_cls), dtype=torch.int, device=self.device)).all() # NOTE
        cls_self_labels = cls_self_mask.nonzero()
        
        cls_self_contrast_scores = torch.matmul(h_cls, h_cls.t())

        # for the diagonal of the contrastive score matrix, to avoid h_cls[i] * h_cls[i], and use h_cls[i] * h_cls_self[i] instead
        h_cls_self = self._get_cell_representation_noised(bert_kwargs)
        cls_self_contrast_scores[cls_self_labels[:, 0], cls_self_labels[:, 1]] = (h_cls * h_cls_self).sum(dim=-1) # positive exists within this batch
    
        cls_contrast_loss = self._ranking_contrastive_loss(cell_type, cls_self_contrast_scores)
        
        return {"loss": cls_contrast_loss}

    def _toall_celltype_contrastive_loss(self, h_cls, h_prot, cell_type,):
        """
        Compute cell type coherence contrastive loss with all available cell types,
            instead of only considering the ones in the current batch.
        """
        
        N_celltype = len(self.ctgraph_sim)
        all_cell_type = torch.arange(N_celltype, device=self.device)
        all_h_prot = self._get_cell_type_representation(all_cell_type) # [N_celltype, dim]

        prot_type_mask = (cell_type.unsqueeze(1) == all_cell_type.unsqueeze(0)).int() # (bsz, N_celltype)
        prot_self_contrast_scores = torch.matmul(h_cls, all_h_prot.t()) # [bsz, N_celltype]
        
        prot_self_labels = prot_type_mask.nonzero()
        prot_self_contrast_scores[prot_self_labels[:, 0], prot_self_labels[:, 1]] = (h_cls * h_prot).sum(dim=-1)

        return prot_self_labels, prot_self_contrast_scores

    def calculate_cell_type_coherence(self, h_cls, cell_type):
        """Compute loss for maintaining cell type coherence in latent space."""
        h_prot = self._get_cell_type_representation(cell_type)
        cls_prot_labels, cls_prot_scores = self._toall_celltype_contrastive_loss(h_cls, h_prot, cell_type)
        
        cls_prot_loss = nn.CrossEntropyLoss()(cls_prot_scores, cls_prot_labels[:, 1]) # same as cls_self_labels
        cls_prot_acc = (cls_prot_scores.argmax(dim=-1) == cls_prot_labels[:, 1]).float().mean()

        reg_loss = F.mse_loss(self.affine_projection(h_prot), h_cls)

        return {
            "loss": cls_prot_loss,
            "acc": cls_prot_acc,
            "reg_loss": reg_loss
        }


    def calculate_losses(self, sequence_output, labels, cell_type, bert_kwargs):

        # masked gene prediction
        masked_gene_info = self.calculate_masked_gene_prediction(sequence_output, labels)
        # cell representation
        h_cls = self._get_cell_representation(sequence_output)
        # relational alignment
        relational_alignment_info = self.calculate_relational_alignment(h_cls, cell_type, bert_kwargs)
        # cell type coherence
        cell_type_coherence_info = self.calculate_cell_type_coherence(h_cls, cell_type)

        # aggregate total loss
        total_loss = (
            masked_gene_info["loss"] 
            + relational_alignment_info["loss"]
            + cell_type_coherence_info["loss"]
            + cell_type_coherence_info["reg_loss"]
        )

        # wandb logging
        if dist.get_rank() == 0 and self.training:
            if self.step_count % (self.total_logging_steps * dist.get_world_size()) == 0:
                metrics = {
                    "train/masked_gene_prediction_loss": masked_gene_info["loss"],
                    "train/masked_gene_prediction_accuracy": masked_gene_info["acc"],
                    "train/relational_alignment_loss": relational_alignment_info["loss"],
                    "train/cell_type_coherence_regulation_loss": cell_type_coherence_info["reg_loss"],
                    "train/cell_type_coherence_loss": cell_type_coherence_info["loss"],
                    "train/cell_type_coherence_accuracy": cell_type_coherence_info["acc"],
                    "train/total_loss": total_loss,
                }
                wandb.log(metrics, commit=False)

        return total_loss, masked_gene_info["prediction_scores"]
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        cell_type: Optional[str] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        if self.training:
            self.step_count += 1
        
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        assert return_dict

        # pass input through the encoder
        bert_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "head_mask": head_mask,
            "inputs_embeds": inputs_embeds,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
        outputs = self.bert(**bert_kwargs)
        sequence_output = outputs[0]
        #ipdb.set_trace()

        if output_hidden_states:
            logits = torch.cat([x[:, 0, :].unsqueeze(1) for x in outputs.hidden_states], dim=1) # [bsz, num_layers, dim]
            return MaskedLMOutput(loss=torch.zeros(1, device=self.device), logits=logits)

        # calculcate all losses
        total_loss, prediction_scores = self.calculate_losses(sequence_output, labels, cell_type, bert_kwargs)

        return MaskedLMOutput(
            loss=total_loss,
            logits=[prediction_scores] if prediction_scores is not None else [],
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class PrototypeContrastiveForSequenceClassification(BertPreTrainedModel):
    
    def __init__(self, config, data_source=None, 
        normalize_flag=False, pass_cell_cls=False,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = PrototypeContrastiveModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

        self.total_logging_steps = config.total_logging_steps
        self.step_count = 0

        self.data_source = data_source
        self.normalize_flag = normalize_flag
        self.pass_cell_cls = pass_cell_cls
        if self.pass_cell_cls:
            self.cell_cls = PrototypeContrastiveOnlyMLMHead(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ) # [bsz, L, dim], [bsz, dim]

        pooled_output = outputs[1]
        if self.pass_cell_cls:
            pooled_output = self.cell_cls(pooled_output)
        if self.normalize_flag:
            pooled_output = F.normalize(pooled_output, p=2, dim=-1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        if dist.get_rank() == 0 and self.training:
            if self.step_count % (self.total_logging_steps * dist.get_world_size()) == 0:
                acc = multiclass_accuracy(logits, labels, num_classes=self.num_labels, average='macro')
                auroc = multiclass_auroc(logits, labels, num_classes=self.num_labels, average='macro')
                macro_f1 = f1_score(logits, labels, task="multiclass", num_classes=self.num_labels, average='macro')

                metrics = {
                    "train/acc": acc, 
                    "train/macro_f1": macro_f1,
                    "train/auroc": auroc,
                }

                metrics = {"/".join(["train"] + [self.data_source] + k.split("/")[1:]): v
                           for k,v in metrics.items()}
                               
                wandb.log(metrics, commit=False)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            #hidden_states=outputs.hidden_states,
            #attentions=outputs.attentions,
        )

