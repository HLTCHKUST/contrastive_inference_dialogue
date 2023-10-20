#!/usr/bin/env python
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import json
from logging import getLogger
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset
from nlgeval.pycocoevalcap.rouge.rouge import Rouge
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.bleu.bleu import Bleu

from src.data_utils.seq2seq_data_utils import Seq2SeqDataset, Seq2SeqDataCollator
from src.data_utils.causal_data_utils import CausalDataset, CausalDataCollator



logger = getLogger(__name__)


def compute_metrics(scorers, predictions, golds, is_glucose_test=False):
    refs, hyps = {}, {}
    task_scores = {}
    for j in range(len(golds)):
        # TODO: this is stupid; need a better way
        if is_glucose_test:
            golds[j] = golds[j][2:-2].replace("'", "").replace('"', '').split(",")
        refs[j] = [golds[j]] if isinstance(golds[j], str) else golds[j]
        hyps[j] = [predictions[j]]

    for scorer, method in scorers:
        score, _ = scorer.compute_score(refs, hyps)
        if isinstance(score, list):
            for m, s in zip(method, score):
                task_scores[m] = round(s, 5)
        else:
            task_scores[method] = round(score, 5)
    return task_scores

def inference(args):
    """Save model.generate results to <out_file>, and return how long it took."""
    if "gpt" in args.model_name_or_path:
        is_decoder_only = True
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side='left')
    else:
        is_decoder_only = False
        add_prefix_space = "bart" in args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, add_prefix_space=add_prefix_space)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(args.device)
    if args.fp16:
        model = model.half()

    lm_datasets = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir)
    
    if is_decoder_only:
        test_dataset = CausalDataset(args, lm_datasets[args.split], tokenizer, predict=True)
    else:
        test_dataset = Seq2SeqDataset(args, lm_datasets[args.split], tokenizer)
    is_glucose_test = args.split == "test" and "glucose" in args.dataset_name

    Path(args.save_path).parent.mkdir(exist_ok=True)

    if args.device == "cpu" and args.fp16:
        # this mix leads to RuntimeError: "threshold_cpu" not implemented for 'Half'
        raise ValueError("Can't mix --fp16 and --device cpu")

    if tokenizer.sep_token is None:
        stop_token = tokenizer.eos_token
    else:
        stop_token = tokenizer.sep_token
    
    print(f"The stop token is {stop_token}")
    print(f"Tokenizer padding side is: {tokenizer.padding_side}")

    loader = DataLoader(
        test_dataset, 
        batch_size=args.bs, 
        sampler=SequentialSampler(test_dataset),
        collate_fn=Seq2SeqDataCollator(tokenizer, args, model.config.decoder_start_token_id) if not is_decoder_only else CausalDataCollator(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        shuffle=False
        )

    generated_sequences, gold_sequences = [], []
    for batch in tqdm(loader, desc='Inference', total=len(loader), ncols=100):
        input_gen_len = batch['input_ids'].shape[1] if is_decoder_only else 0
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        
        gen_kwargs = {
            "top_k": args.k,
            "top_p": args.p,
            "do_sample": args.sampling,
            "pad_token_id": tokenizer.pad_token_id,
            "num_beams": args.n_beam, 
            "temperature": args.temperature,
            "max_length": args.max_target_length + input_gen_len, 
            "min_length": 5,
            "repetition_penalty": args.repetition_penalty,
        }
        
        if "token_type_ids" in batch:
            token_type_ids = batch["token_type_ids"]
            gen_kwargs.update({"token_type_ids": token_type_ids.to(args.device)})

        if not is_decoder_only:
            gen_kwargs.update({"decoder_start_token_id": tokenizer.bos_token_id})

        # print(tokenizer.batch_decode(input_ids))
        # print("attention_mask", attention_mask)
        # generated_sequence = torch.ones_like(input_ids)
        generated_sequence = model.generate(
            input_ids=input_ids.to(args.device),
            attention_mask=attention_mask.to(args.device),
            **gen_kwargs,
        )
        # print("generated_sequence", tokenizer.batch_decode(generated_sequence))
        for generated_sequence, response in zip(generated_sequence[:, input_gen_len:], batch["labels"]):
            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True, skip_special_tokens=True) # DO NOT skip_special_tokens
            text = text[: text.find(stop_token) if stop_token and text.find(stop_token)>0 else None]
            generated_sequences.append(text)
            
            response = response[response != -100]
            response_text = tokenizer.decode(response, clean_up_tokenization_spaces=True, skip_special_tokens=True)

            gold_sequences.append(response_text)

        if args.debug:
            print(f"The generated sentence is: {text}")
            print(f"The golden sentence is: {response_text}")
            print("="*80)
            input()
            

    core_filename = f"{args.save_path}/{args.split}"
        
    golds_filename = core_filename + "_gold.txt"
    generations_filename = core_filename + "_generation.txt"
    scores_filename = core_filename + "_scores.json"
        
    with open(generations_filename, "w") as f:
        for line in generated_sequences:
            f.write(line.replace("\n", " ").replace(tokenizer.pad_token, "").strip()+"\n")
    with open(golds_filename, "w") as f:
        for line in gold_sequences:
            f.write(line.replace("\n", " ").replace(tokenizer.pad_token, "").strip()+"\n")
    

    # Let's compute scores!
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    scores = compute_metrics(scorers, generated_sequences, gold_sequences, is_glucose_test)
    scores['sample_size'] = len(lm_datasets[args.split])

    keys, values = [], []
    for k,v in scores.items():
        keys.append(k)
        values.append(str(round(v*100,2)))
    print(" ".join(keys))
    print(" ".join(values))
        
    with open(scores_filename, "w") as f:
        json.dump(scores, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--save_path", type=str, help="where to save generations")
    parser.add_argument('--max_source_length', type=int, default=768) 
    parser.add_argument('--max_target_length', type=int, default=128) 
    parser.add_argument(
        "--prefix", type=str, required=False, default=None, help="will be added to the begininng of src examples"
    )
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument('--dataset_name', type=str, default="src/data_utils/cicero.py")
    parser.add_argument('--dataset_config_name', type=str, default="cicero_nlg")
    parser.add_argument('--split', choices=["train", "validation", "test"], default="test")
    parser.add_argument('--question_type', type=str, default=None, help="choose from m,s,l,a,c and its combination. \
                                                                         inputting msl means you're choosing m,s,l,ms,sl,ml,msl")
    parser.add_argument(
        "--preprocessing_num_workers",
        default=None,
        type=int,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--cache_dir", default=None, help="Where do you want to store the pretrained models downloaded from huggingface.co")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument('-cu', '--cuda', help='Cude device number', type=str, required=False, default='0')
    parser.add_argument("--seed", help="random seed", default=42, type=int, required=False)
    # generation settings
    parser.add_argument("--n_beam", type=int, default=5, help="beam size")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--debug", action="store_true", help="Enter DEBUG mode")
    parser.add_argument("--add_bos", action="store_true", help="Whether to add additional start of generation token for GPT-2.")
    parser.add_argument("--add_eos_during_generation", action="store_true", help="Whether to add additional eos token at the end of the input during generation.")
    
    # specifically for llama
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_weights", type=str, help="path to the model trained with lora")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    set_seed(args.seed)

    inference(args)