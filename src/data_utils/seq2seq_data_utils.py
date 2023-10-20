from typing import Dict

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import T5Tokenizer, T5TokenizerFast
from transformers.models.bart.modeling_bart import shift_tokens_right


def load_datasets(data_args, cache_dir):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=cache_dir)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files, field="data")
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    return datasets


def preprocess(data_args, datasets, tokenizer):
    if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
        train_dataset = datasets["train"].select(range(data_args.max_train_samples))
    else:
        train_dataset = datasets["train"]
    train_dataset = Seq2SeqDataset(data_args, train_dataset, tokenizer)

    if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
        validation_dataset = datasets["validation"].select(range(data_args.max_eval_samples))
    else:
        validation_dataset = datasets["validation"]
    validation_dataset = Seq2SeqDataset(data_args, validation_dataset, tokenizer)
    
    if data_args.max_predict_samples is not None and data_args.max_predict_samples > 0:
        test_dataset = datasets["test"].select(range(data_args.max_predict_samples))
    else:
        test_dataset = datasets["test"]
    test_dataset = Seq2SeqDataset(data_args, test_dataset, tokenizer)

    return train_dataset, validation_dataset, test_dataset


class Seq2SeqDataset(Dataset):
    """A dataset that calls prepare_seq2seq_batch."""
    def __init__(self, data_args, data, tokenizer):
        self.args = data_args
        self.data = data
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.eos_token if tokenizer.sep_token is None else tokenizer.sep_token
        self.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token is None else tokenizer.pad_token_id
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        return self.tokenize_function(self.data[index])

    def encode_line(self, line, max_length, pad_to_max_length=True, return_tensors="pt"):
        if not isinstance(line, list):
            line = [line]

        return self.tokenizer(
            line,
            max_length=max_length,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
        )

    def tokenize_function(self, example):
        SEP_TOKEN = self.sep_token  # in CICERO original code, SEP_TOKEN is "\\n"
        src = SEP_TOKEN.join([example["question"], \
                                "target: " +  example["target"], \
                                "context: " + example["dialogue"]])

        tgt = example["answer"]

        # print("src:", src)
        # print("tgt:", tgt)
        source = self.encode_line(src, self.args.max_source_length)
        target = self.encode_line(tgt, self.args.max_target_length)

        # Tokenize negative samples
        if "negative_samples" in example:
            negative_samples = self.encode_line(example["negative_samples"], self.args.max_target_length)

        source_ids = source["input_ids"].squeeze()
        target_ids = target["input_ids"].squeeze()
        attention_mask = source["attention_mask"].squeeze()

        input_ids = torch.LongTensor(source_ids)
        attention_mask = torch.LongTensor(attention_mask)
        labels = torch.LongTensor(target_ids)
        label_pad_mask = labels.eq(self.tokenizer.pad_token_id)
        labels[label_pad_mask.bool()] = -100

        if "negative_samples" in example:
            negative_sample_ids = torch.LongTensor(negative_samples["input_ids"])
            negative_pad_mask = negative_sample_ids.eq(self.tokenizer.pad_token_id)
            negative_sample_ids[negative_pad_mask.bool()] = -100
            return {"input_ids":input_ids, "attention_mask":attention_mask, "labels": labels, "negative_ids": negative_sample_ids}
        else:
            return {"input_ids":input_ids, "attention_mask":attention_mask, "labels": labels}


class Seq2SeqDataCollator:
    def __init__(self, tokenizer, data_args, decoder_start_token_id):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args
        # self.dataset_kwargs = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] if torch.is_tensor(x["input_ids"]) else torch.LongTensor(x["input_ids"]) for x in batch])
        attention_mask = torch.stack([x["attention_mask"] if torch.is_tensor(x["attention_mask"]) else torch.LongTensor(x["attention_mask"]) for x in batch])
        labels = torch.stack([x["labels"] if torch.is_tensor(x["labels"]) else torch.LongTensor(x["labels"]) for x in batch])

        if "negative_ids" in batch[0]:
            negative_ids = torch.stack([x["negative_ids"] if torch.is_tensor(x["negative_ids"]) else torch.LongTensor(x["negative_ids"]) for x in batch])
            num_negative_ids = negative_ids.shape[1]
            
            labels = labels.unsqueeze(1)
            # Cat the labels for original samples and negative samples together to ensure them in the same seq length after `trim_batch`
            labels = torch.cat((labels, negative_ids), axis=1)
            # Reshape labels for `trim_batch`
            labels_shape = labels.shape
            labels = labels.view(-1, labels_shape[2])
        
        labels = trim_batch(labels, -100)

        input_ids, attention_mask = trim_batch(input_ids, self.pad_token_id, attention_mask=attention_mask)

        if isinstance(self.tokenizer, T5Tokenizer) or isinstance(self.tokenizer, T5TokenizerFast):
            decoder_input_ids = self._shift_right_t5(labels)
        else: # shift_tokens_right function is taken from BART
            decoder_input_ids = shift_tokens_right(labels, self.pad_token_id, self.decoder_start_token_id)
        decoder_input_ids.masked_fill_(decoder_input_ids == -100, self.pad_token_id)

        if "negative_ids" in batch[0]:
            # Reshape labels back
            labels            = labels.view(labels_shape[0], labels_shape[1], -1)
            decoder_input_ids = decoder_input_ids.view(labels_shape[0], labels_shape[1], -1)
            # Separate the labels for original samples and negative samples 
            # negative_labels = labels[:, -num_negative_ids:, :].clone()
            # labels          = labels[:, 0, :].squeeze()
            # negative_ids    = decoder_input_ids[:, -num_negative_ids:, :].clone()
            # decoder_input_ids = decoder_input_ids[:, 0, :].squeeze()
            negative_ids = None
            
            gt_labels = labels[:, 0, :].clone().unsqueeze(1).expand(-1, labels_shape[1], -1)
            negative_mask = torch.ones_like(labels).masked_fill(labels == gt_labels, 0)
            negative_mask = negative_mask.masked_fill(labels == self.pad_token_id, 0)

        new_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }

        if "negative_ids" in batch[0]:     
            new_batch.update({
                "negative_ids": negative_ids,
                "negative_nums": num_negative_ids,
                "decoder_negative_mask": negative_mask,
            })   

        return new_batch

    def _shift_right_t5(self, input_ids):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.decoder_start_token_id
        return shifted_input_ids

    
def trim_batch(input_ids, pad_token_id, attention_mask=None):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])
    