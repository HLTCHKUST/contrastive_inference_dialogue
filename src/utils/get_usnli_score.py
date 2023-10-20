# modified version of https://github.com/etsukokuste/CoINF/blob/e992d02f662959cf5ea90cd8b41582bbb11b630c/metrics/anli/evaluate.py
import os
import json
import argparse
from typing import Optional, Union, Dict

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from datasets import load_dataset


def trim_batch(input_ids, pad_token_id, attention_mask=None, token_type_ids=None):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None and token_type_ids is None:
        return input_ids[:, keep_column_mask]
    elif token_type_ids is None:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask], token_type_ids[:, keep_column_mask])
    

class NLIDataset(Dataset):
    def __init__(self, args, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token is None else tokenizer.pad_token_id
        self.args = args
    
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
        tokenized_line = self.encode_line(line, self.args.max_length)
        input_ids = tokenized_line["input_ids"].squeeze()
        attention_mask = tokenized_line["attention_mask"].squeeze()
        token_type_ids = tokenized_line["token_type_ids"].squeeze()
        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        token_type_ids = torch.LongTensor(token_type_ids)
        return {"input_ids":input_ids, "attention_mask":attention_mask, "token_type_ids":token_type_ids, "labels":example["label"]}


class NLIDataCollator:
    def __init__(
            self, 
            tokenizer: PreTrainedTokenizerBase,
            padding: Union[bool, str, PaddingStrategy] = True,
            max_length: Optional[int] = None,
        ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
    
    def __call__(self, batch):
        input_ids = torch.stack([x["input_ids"] if torch.is_tensor(x["input_ids"]) else torch.LongTensor(x["input_ids"]) for x in batch])
        attention_mask = torch.stack([x["attention_mask"] if torch.is_tensor(x["attention_mask"]) else torch.LongTensor(x["attention_mask"]) for x in batch])
        token_type_ids = torch.stack([x["token_type_ids"] if torch.is_tensor(x["token_type_ids"]) else torch.LongTensor(x["token_type_ids"]) for x in batch])
        input_ids, attention_mask, token_type_ids = trim_batch(input_ids, self.tokenizer.pad_token_id, attention_mask=attention_mask, token_type_ids=token_type_ids)
        labels = [x["labels"] for x in batch]
        new_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
        return new_batch


def main(args):
    id2label = {
            0: "entailment",
            1: "neutral",
            2: "contradiction",
        }
    
    print('Loading tokenizer and model...')

    num_labels = 1 if "usnli" in args.model_name_or_path else 3
    if args.model_name_or_path == "/nfs/etsuko/CoINF/save/roberta-large-anli/":
        tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        state_dict = torch.load(f"{args.model_name_or_path}/model.pt")
        model = AutoModelForSequenceClassification.from_pretrained("roberta-large", state_dict=state_dict, num_labels=num_labels).to(args.device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels).to(args.device)

    lm_dataset = load_dataset(args.dataset_name)[args.split]

    test_dataset = NLIDataset(args, lm_dataset, tokenizer)
    loader = DataLoader(
        test_dataset,
        batch_size=args.bs,
        sampler=SequentialSampler(test_dataset),
        collate_fn=NLIDataCollator(tokenizer, args.max_length),
        shuffle=False
    )

    gold_labels = []
    predicted_probs = []
    for batch in tqdm(loader):
        input_ids, attention_mask, token_type_ids = batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
        outputs = model(
            input_ids=input_ids.to(args.device),
            attention_mask=attention_mask.to(args.device),
            token_type_ids=token_type_ids.to(args.device),
            labels=None,
        )
        if num_labels == 3:
            predicted_probability = torch.softmax(outputs.logits, dim=1)
            predicted_index = torch.argmax(predicted_probability, dim=1).detach().tolist()
            predicted_labels += predicted_index
            predicted_probs += predicted_probability.detach().tolist()
        else:
            predicted_probs += outputs.logits.detach().tolist()
            gold_labels += batch["labels"]

    # average prediction
    if num_labels == 3:
        avg_predicted_probs = [sum(x) for x in zip(*predicted_probs)]
        avg_predicted_probs = [x/len(lm_dataset) for x in avg_predicted_probs]
        print("entail:", round(avg_predicted_probs[0], 4), "neutral:", round(avg_predicted_probs[1], 4), "contradict:", round(avg_predicted_probs[2], 4))
        results = {
            "average": avg_predicted_probs,
            "predicted_labels": predicted_labels,
            "predicted_probs": predicted_probs
            }
    else:
        # flatten scores
        predicted_probs = [inner for outer in predicted_probs for inner in outer]
        ent_score = [p for p, l in zip(predicted_probs, gold_labels) if l == 0]
        neu_score = [p for p, l in zip(predicted_probs, gold_labels) if l == 1]
        con_score = [p for p, l in zip(predicted_probs, gold_labels) if l == 2]
        avg_ent = sum(ent_score) / len(ent_score)
        avg_neu = sum(neu_score) / len(neu_score)
        avg_con = sum(con_score) / len(con_score)
        print("ENT:", avg_ent, len(ent_score), "NEU:", avg_neu, len(neu_score), "CON:", avg_con, len(con_score))
        results = {
            "average_entailment": avg_ent,
            "average_neutral": avg_neu,
            "average_contradict": avg_con,
            "predicted_probs": predicted_probs,
            "gold_labels": gold_labels,
            }
    
    save_file_name = f"{args.save_path}/snli_{args.split}_usnli_scores.json" 
    with open(save_file_name, "w") as f:
        json.dump(results, f)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="save/roberta-large-anli-usnli/",
        required=False,
        help="Pre-trained model name or path")
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to file containing list of hypotheses")
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        required=False,
        help="Maximum length")
    parser.add_argument('-cu', '--cuda', help='Cude device number', type=str, required=False, default='0')
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument("--split", type=str, default="validation", required=False)
    parser.add_argument("--dataset_name", choices=["snli", "anli", "multi_nli"], default="snli")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    main(args)