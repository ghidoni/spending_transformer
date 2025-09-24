
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, SequenceClassifierOutputWithPast

class NuFormerConfig(PretrainedConfig):
    model_type = "nuformer"

    def __init__(
        self,
        vocab_size=1000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        num_labels=2,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.num_labels = num_labels
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

class NuFormerForCausalLM(PreTrainedModel):
    config_class = NuFormerConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads),
            num_layers=config.num_hidden_layers
        )
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        embedding_output = self.embedding(input_ids)
        decoder_output = self.transformer(embedding_output, embedding_output) # Self-attention
        logits = self.lm_head(decoder_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, self.config.vocab_size)
            labels = labels.view(-1)
            loss = loss_fct(logits, labels)

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
        )

class NuFormerForSequenceClassification(PreTrainedModel):
    config_class = NuFormerConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer = NuFormerForCausalLM(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask, labels=labels)
        pooled_logits = outputs.logits.mean(dim=1)
        logits = self.score(pooled_logits)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
        )
