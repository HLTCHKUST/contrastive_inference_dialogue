from typing import Dict

import torch
from torch.utils.data import Dataset


def preprocess(training_args, data_args, datasets, tokenizer):
    if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
        train_dataset = datasets["train"].select(range(data_args.max_train_samples))
    else:
        train_dataset = datasets["train"]
    train_dataset = NLIDataset(data_args, train_dataset, tokenizer)

    if training_args.do_eval:
        if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
            validation_dataset = datasets["validation"].select(range(data_args.max_eval_samples))
        else:
            validation_dataset = datasets["validation"]
        validation_dataset = NLIDataset(data_args, validation_dataset, tokenizer)
    else:
        validation_dataset = None
    
    if training_args.do_predict:
        if data_args.max_predict_samples is not None and data_args.max_predict_samples > 0:
            test_dataset = datasets["test"].select(range(data_args.max_predict_samples))
        else:
            test_dataset = datasets["test"]
        test_dataset = NLIDataset(data_args, test_dataset, tokenizer)
    else:
        test_dataset = None

    return train_dataset, validation_dataset, test_dataset


class NLIDataset(Dataset):
    def __init__(self, data_args, data, tokenizer):
        self.args = data_args
        self.data = data
        self.tokenizer = tokenizer
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
            return_token_type_ids=True,
        )

    def tokenize_function(self, example):
        line = (example["premise"], example["hypothesis"])
        tokenized_line = self.encode_line(line, self.args.max_seq_length)
        input_ids = tokenized_line["input_ids"].squeeze()
        attention_mask = tokenized_line["attention_mask"].squeeze()
        token_type_ids = tokenized_line["token_type_ids"].squeeze()
        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        token_type_ids = torch.LongTensor(token_type_ids)
        labels = None
        if self.args.dataset_config_name == "usnli":
            labels = torch.FloatTensor([example["label"]])
        elif self.args.dataset_config_name == "joci":
            labels = torch.LongTensor([example["label"]])
        return {"input_ids":input_ids, "attention_mask":attention_mask, "token_type_ids":token_type_ids, "labels":labels}


class NLIDataCollator:
    def __init__(
            self, tokenizer, data_args,
        ):
        self.tokenizer = tokenizer
        self.args = data_args
    
    def __call__(self, batch):
        input_ids = torch.stack([x["input_ids"] if torch.is_tensor(x["input_ids"]) else torch.LongTensor(x["input_ids"]) for x in batch])
        attention_mask = torch.stack([x["attention_mask"] if torch.is_tensor(x["attention_mask"]) else torch.LongTensor(x["attention_mask"]) for x in batch])
        token_type_ids = torch.stack([x["token_type_ids"] if torch.is_tensor(x["token_type_ids"]) else torch.LongTensor(x["token_type_ids"]) for x in batch])

        if self.args.dataset_config_name == "usnli":
            labels = torch.stack([x["labels"] if torch.is_tensor(x["labels"]) else torch.FloatTensor(x["labels"]) for x in batch])
        elif self.args.dataset_config_name == "joci":
            labels = torch.stack([x["labels"] if torch.is_tensor(x["labels"]) else torch.LongTensor(x["labels"]) for x in batch])
        else:
            labels = None
        input_ids, attention_mask, token_type_ids = trim_batch(input_ids, self.tokenizer.pad_token_id, attention_mask=attention_mask, token_type_ids=token_type_ids)
        new_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
        return new_batch


def trim_batch(input_ids, pad_token_id, attention_mask=None, token_type_ids=None):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None and token_type_ids is None:
        return input_ids[:, keep_column_mask]
    elif token_type_ids is None:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask], token_type_ids[:, keep_column_mask])