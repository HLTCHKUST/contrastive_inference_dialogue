from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from src.utils.utils import START_OF_GEN_TOKEN


def preprocess_causal(data_args, datasets, tokenizer):
    if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
        train_dataset = datasets["train"].select(range(data_args.max_train_samples))
    else:
        train_dataset = datasets["train"]
    train_dataset = CausalDataset(data_args, train_dataset, tokenizer)

    if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
        validation_dataset = datasets["validation"].select(range(data_args.max_eval_samples))
    else:
        validation_dataset = datasets["validation"]
    validation_dataset = CausalDataset(data_args, validation_dataset, tokenizer)
    
    if data_args.max_predict_samples is not None and data_args.max_predict_samples > 0:
        test_dataset = datasets["test"].select(range(data_args.max_predict_samples))
    else:
        test_dataset = datasets["test"]
    test_dataset = CausalDataset(data_args, test_dataset, tokenizer, predict=True)

    return train_dataset, validation_dataset, test_dataset


class CausalDataset(Dataset):
    """A dataset that calls prepare_seq2seq_batch."""
    def __init__(self, data_args, data, tokenizer, predict=False):
        self.args = data_args
        self.data = data
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.eos_token if tokenizer.sep_token is None else tokenizer.sep_token
        self.eos_token = tokenizer.eos_token
        self.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        self.tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
        self.predict = predict
        
        if self.sep_token == " " or self.sep_token == "" or self.sep_token is None:
            self.sep_token = "\n\n"
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        return self.tokenize_function(self.data[index])
    
    def tokenize(self, line, max_length, add_eos_token=True):
        tokenized_line = self.tokenizer(
            line,
            max_length=max_length,
            padding=False,
            truncation=True,
            return_tensors=None,
        )
        
        if (
            tokenized_line["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(tokenized_line["input_ids"]) < max_length
            and add_eos_token
        ):
            tokenized_line["input_ids"].append(self.tokenizer.eos_token_id)
            tokenized_line["attention_mask"].append(1)

        tokenized_line["labels"] = tokenized_line["input_ids"].copy()

        return tokenized_line
    
    def encode_line(self, src_line, tgt_line, max_length, train_on_inputs=False):
        if self.predict:
            full_line = src_line + self.sep_token + "answer:" if not self.args.add_bos else src_line + START_OF_GEN_TOKEN
            full_tokenized_line = self.tokenize(full_line, max_length, add_eos_token=self.args.add_eos_during_generation)
            
            label_tokenized_line = self.tokenize(tgt_line, max_length)
            full_tokenized_line["labels"] = label_tokenized_line["input_ids"]
        else:
            full_line = src_line + self.sep_token + "answer: " + tgt_line if not self.args.add_bos else src_line + START_OF_GEN_TOKEN + tgt_line
            full_tokenized_line = self.tokenize(full_line, max_length)
            if not train_on_inputs:
                input_line = src_line + self.sep_token + "answer:" if not self.args.add_bos else src_line + START_OF_GEN_TOKEN
                input_tokenized_line = self.tokenize(input_line, max_length, add_eos_token=False)
                input_len = len(input_tokenized_line["input_ids"])
                full_tokenized_line["labels"] = [
                    -100
                ] * input_len + full_tokenized_line["labels"][
                    input_len:
                ]  # could be sped up, probably
        return full_tokenized_line
    
    def pad_seqs_for_contrast(self, seqs, pad_token_id=0):
        max_length = max([len(l) for l in seqs])
        for i in range(len(seqs)):
            remainder = [pad_token_id] * (max_length - len(seqs[i]))
            seqs[i] = remainder + seqs[i]
        return seqs
    

    def tokenize_function(self, example):
        SEP_TOKEN = self.sep_token  # in CICERO original code, SEP_TOKEN is "\\n"
        src = SEP_TOKEN.join([example["question"], \
                                "target: " +  example["target"], \
                                "context: " + example["dialogue"]])

        tgt = example["answer"]

        source = self.encode_line(src, tgt, self.args.max_source_length)

        input_ids = source["input_ids"]
        attention_mask = source["attention_mask"]
        labels = source["labels"]

        if "negative_samples" in example:
            input_ids = [input_ids]
            attention_mask = [attention_mask]
            labels = [labels]
            # Tokenize negative samples
            for s in example["negative_samples"]:
                encoded_s = self.encode_line(src, s, self.args.max_source_length)
                input_ids.append(encoded_s["input_ids"])
                attention_mask.append(encoded_s["attention_mask"])
                labels.append(encoded_s["labels"])

            return {"input_ids":input_ids, "attention_mask":attention_mask, "labels": labels, "negative_nums": len(example["negative_samples"])}
        else:
            return {"input_ids":input_ids, "attention_mask":attention_mask, "labels": labels}


@dataclass
class CausalDataCollator:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*
            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        negative_nums = [f.pop("negative_nums") for f in features][0] if "negative_nums" in features[0].keys() else None
        
        if negative_nums is not None:
            # data collator for contrastive learning settings
            # merge features for different data samples
            new_features = defaultdict(list)
            bsz = len(features)
            for feature in features:
                for k, v in feature.items():
                    new_features[k].extend(v)
            features = new_features
        
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            if negative_nums is not None:
                labels = features["labels"]
                max_label_length = max(len(l) for l in labels)
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                
                for i in range(len(labels)):
                    remainder = [self.label_pad_token_id] * (max_label_length - len(features["labels"][i]))
                    if isinstance(features["labels"][i], list):
                        features["labels"][i] = (
                            features["labels"][i] + remainder if padding_side == "right" else remainder + features["labels"][i]
                        )
                    elif padding_side == "right":
                        features["labels"][i] = np.concatenate([features["labels"][i], remainder]).astype(np.int64)
                    else:
                        features["labels"][i] = np.concatenate([remainder, features["labels"][i]]).astype(np.int64)

            else:

                max_label_length = max(len(l) for l in labels)
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
            
                for feature in features:
                    remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                    if isinstance(feature["labels"], list):
                        feature["labels"] = (
                            feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                        )
                    elif padding_side == "right":
                        feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                    else:
                        feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        # print("features")
        # for key in features:
        #     print(key, features[key].shape)
        # input()
        if negative_nums is not None:
            features["negative_nums"] = negative_nums
        return features