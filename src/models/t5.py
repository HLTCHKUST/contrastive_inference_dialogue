# modified from https://github.com/seanie12/CLAPS/blob/a0a5747eb2e967d2828fd68683f8a325f7abbe31/src/summarization/models.py
# and huggingface https://github.com/huggingface/transformers/blob/dcb08b99f44919425f8ba9be9ddcc041af8ec25e/src/transformers/models/t5/modeling_t5.py
import copy
from typing import Optional, Union, Tuple
import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.models.t5.modeling_t5 import (
    T5PreTrainedModel,
    T5Config, 
    T5Stack, 
    T5_START_DOCSTRING, 
    PARALLELIZE_DOCSTRING,
    DEPARALLELIZE_DOCSTRING,
    _CONFIG_FOR_DOC,
    T5_INPUTS_DOCSTRING,
    __HEAD_MASK_WARNING_MSG)
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.utils import logging, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings, ModelOutput


logger = logging.get_logger(__name__)


@dataclass
class Seq2SeqLMOutputWithContrast(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    inbatch_loss: Optional[torch.FloatTensor] = None
    insample_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


@add_start_docstrings("""T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING)
class T5ForConditionalGenerationWithContrastiveLoss(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.projection_hidden_size > 0:
            self.projection = nn.Sequential(nn.Linear(config.projection_hidden_size, config.projection_hidden_size), nn.ReLU())
        else:
            self.projection = None
        self.tau_sample = config.tau_sample
        self.tau_batch = config.tau_batch
        self.coef_inbatch = config.coef_inbatch
        self.coef_insample = config.coef_insample

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.projection = self.projection.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.projection = self.projection.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutputWithContrast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        negative_ids: Optional[torch.LongTensor] = None,
        negative_nums: Optional[int] = 0,
        decoder_negative_mask: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutputWithContrast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        Returns:
        Examples:
        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")
        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        bsz = hidden_states.shape[0]

        if negative_nums > 0:
            # if negative_ids is None, then it means that the negative samples are already concatenated 
            # with `decoder_input_ids` and `labels`.
            bsz = hidden_states.shape[0]
            # Expand encoder hidden states from [bsz, seq_len, hdim] to [bsz, 1+num_negative, seq_len, hdim]
            expand_size = negative_nums + 1
            hidden_states = hidden_states.unsqueeze(1).expand(-1, expand_size, -1, -1)
            attention_mask = attention_mask.unsqueeze(1).expand(-1, expand_size, -1)

            # Reshape hidden states from [bsz, 1+num_negatives, seq_len, hdim] to [bsz*(1+num_negatives), seq_len, hdim]
            hidden_states = hidden_states.reshape(bsz * expand_size, hidden_states.shape[2], hidden_states.shape[3])
            # Reshape attention mas kand decoder input ids from [bsz, 1+num_negatives, seq_len] to [bsz*(1+num_negatives), seq_len]
            attention_mask = attention_mask.reshape(bsz * expand_size, -1)
            decoder_input_ids = decoder_input_ids.reshape(bsz * expand_size, -1)


        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        mix_logits = self.lm_head(sequence_output)

        if negative_nums > 0:
            # Reshape from [bsz*(1+num_negative), seq_len, vocab_size] to [bsz, 1+num_negative, seq_len, vocab_size]
            mix_logits = mix_logits.view(bsz, expand_size, mix_logits.shape[1], -1)

        loss, lm_loss, inbatch_loss, insample_loss = None, None, None, None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # Language modeling loss
            # Take the first index of the second dimension of both `mix_logits` and `labels`
            # We denote the logits and label above as `lm_logits` and `lm_labels`
            if negative_nums > 0:
                lm_logits = mix_logits[:, 0, :, :].squeeze(1)
                lm_labels = labels[:, 0, :].squeeze(1)
            else:
                lm_logits = mix_logits
                lm_labels = labels
            loss = loss_fct(lm_logits.reshape(-1, lm_logits.size(-1)), lm_labels.reshape(-1))
            lm_loss = loss.item()

            if self.coef_inbatch > 0 or self.coef_insample > 0:
                # forward to get projection
                proj_enc_h = self.projection(hidden_states)
                proj_dec_h = self.projection(sequence_output)
                # get loss for within-batch negative samples
                avg_enc = self.avg_pool(proj_enc_h, attention_mask)
                avg_dec = self.avg_pool(proj_dec_h, decoder_input_ids)
                loss_contrastive, inbatch_loss, insample_loss = self.get_contrastive_loss(avg_enc, avg_dec, negative_nums=negative_nums, bsz=bsz)
                loss += loss_contrastive
            
            inbatch_loss  = inbatch_loss.item() if inbatch_loss is not None else None
            insample_loss = insample_loss.item() if insample_loss is not None else None

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss, lm_loss, inbatch_loss, insample_loss,) + output) if loss is not None else output

        return Seq2SeqLMOutputWithContrast(
            loss=loss,
            lm_loss=lm_loss,
            inbatch_loss=inbatch_loss,
            insample_loss=insample_loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
    def get_contrastive_loss(self, avg_enc, avg_dec, negative_nums, bsz):
        cos = nn.CosineSimilarity(dim=-1)
        cont_crit = nn.CrossEntropyLoss()
        device = avg_enc.device
        if negative_nums > 0:
            avg_enc = avg_enc.view(bsz, negative_nums+1, avg_enc.shape[1])
            avg_dec = avg_dec.view(bsz, negative_nums+1, avg_dec.shape[1])
        if self.coef_inbatch > 0:
            if negative_nums > 0:
                sim_matrix_inbatch = cos(avg_enc[:, 0, :].unsqueeze(1), avg_dec[:, 0, :].unsqueeze(0)) / self.tau_batch
            else:
                sim_matrix_inbatch = cos(avg_enc.unsqueeze(1), avg_dec.unsqueeze(0)) / self.tau_batch
            labels_inbatch = torch.arange(bsz, device=device)
            cont_loss_inbatch = self.coef_inbatch * cont_crit(sim_matrix_inbatch, labels_inbatch)
        else:
            cont_loss_inbatch = None

        cont_loss = cont_loss_inbatch if cont_loss_inbatch is not None else 0
        if negative_nums > 0 and self.coef_insample > 0:
            sim_matrix_sample = cos(avg_enc, avg_dec) / self.tau_sample
            labels_sample = torch.zeros([bsz]).type(torch.LongTensor).to(device)
            cont_loss_insample = cont_crit(sim_matrix_sample, labels_sample)
            cont_loss  += self.coef_insample * cont_loss_insample
        else:
            cont_loss_insample = None
        return cont_loss, cont_loss_inbatch, cont_loss_insample

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / length

        return avg_hidden
    
    def forward_for_projection(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        negative_ids: Optional[torch.LongTensor] = None,
        negative_nums: Optional[int] = 0,
    ) -> torch.LongTensor:
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        bsz = hidden_states.shape[0]

        if negative_nums > 0:
            # if negative_ids is None, then it means that the negative samples are already concatenated 
            # with `decoder_input_ids` and `labels`.
            bsz = hidden_states.shape[0]
            # Expand encoder hidden states from [bsz, seq_len, hdim] to [bsz, 1+num_negative, seq_len, hdim]
            expand_size = negative_nums + 1
            hidden_states = hidden_states.unsqueeze(1).expand(-1, expand_size, -1, -1)
            attention_mask = attention_mask.unsqueeze(1).expand(-1, expand_size, -1)

            # Reshape hidden states from [bsz, 1+num_negatives, seq_len, hdim] to [bsz*(1+num_negatives), seq_len, hdim]
            hidden_states = hidden_states.reshape(bsz * expand_size, hidden_states.shape[2], hidden_states.shape[3])
            # Reshape attention mas kand decoder input ids from [bsz, 1+num_negatives, seq_len] to [bsz*(1+num_negatives), seq_len]
            attention_mask = attention_mask.reshape(bsz * expand_size, -1)
            decoder_input_ids = decoder_input_ids.reshape(bsz * expand_size, -1)


        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        # forward to get projection
        if negative_nums <= 0:
            raise NotImplementedError("embedding comparison only supports datasets with negative samples")
        
        proj_enc_h = self.projection(hidden_states)
        proj_dec_h = self.projection(sequence_output)
        avg_enc = self.avg_pool(proj_enc_h, attention_mask)
        avg_dec = self.avg_pool(proj_dec_h, decoder_input_ids)
        # get similarity of embeddings for in-sample
        cos = nn.CosineSimilarity(dim=-1)
        avg_enc = avg_enc.view(bsz, negative_nums+1, avg_enc.shape[1])
        avg_dec = avg_dec.view(bsz, negative_nums+1, avg_dec.shape[1])
        sim_matrix_sample = cos(avg_enc, avg_dec) / self.tau_sample
        predicted_rank = torch.argsort(sim_matrix_sample, dim=1, descending=True)
        predicted_rank = predicted_rank.detach().cpu()
        return predicted_rank

    def forward_for_ppl(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        negative_ids: Optional[torch.LongTensor] = None,
        negative_nums: Optional[int] = 0,
    ) -> torch.LongTensor:
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        bsz = hidden_states.shape[0]

        if negative_nums > 0:
            # if negative_ids is None, then it means that the negative samples are already concatenated 
            # with `decoder_input_ids` and `labels`.
            bsz = hidden_states.shape[0]
            # Expand encoder hidden states from [bsz, seq_len, hdim] to [bsz, 1+num_negative, seq_len, hdim]
            expand_size = negative_nums + 1
            hidden_states = hidden_states.unsqueeze(1).expand(-1, expand_size, -1, -1)
            attention_mask = attention_mask.unsqueeze(1).expand(-1, expand_size, -1)

            # Reshape hidden states from [bsz, 1+num_negatives, seq_len, hdim] to [bsz*(1+num_negatives), seq_len, hdim]
            hidden_states = hidden_states.reshape(bsz * expand_size, hidden_states.shape[2], hidden_states.shape[3])
            # Reshape attention mas kand decoder input ids from [bsz, 1+num_negatives, seq_len] to [bsz*(1+num_negatives), seq_len]
            attention_mask = attention_mask.reshape(bsz * expand_size, -1)
            decoder_input_ids = decoder_input_ids.reshape(bsz * expand_size, -1)


        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        # forward to get projection
        if negative_nums <= 0:
            raise NotImplementedError("embedding comparison only supports datasets with negative samples")

        mix_logits = self.lm_head(sequence_output)
        mix_logits = mix_logits.view(bsz, expand_size, mix_logits.shape[1], -1)

        loss_fnc = CrossEntropyLoss(ignore_index=-100, reduction="none")
        loss = loss_fnc(mix_logits.reshape(-1, mix_logits.size(-1)), labels.reshape(-1))
        loss = loss.reshape(labels.shape)
        loss = torch.mean(loss, dim=-1)
        predicted_rank = torch.argsort(loss, dim=1, descending=False)
        predicted_rank = predicted_rank.detach().cpu()
        return predicted_rank