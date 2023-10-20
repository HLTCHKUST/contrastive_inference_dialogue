import os
import json
import random
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset
from tqdm import tqdm
from nlgeval.pycocoevalcap.rouge.rouge import Rouge
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.bleu.bleu import Bleu

from src.utils.cicero_prompt import SUBSEQ_EVENT_TEMPLATE, CAUSE_TEMPLATE, PREREQUISITE_TEMPLATE, REACTION_TEMPLATE, MOTIVATION_TEMPLATE


EXAMPLE_SPLITTER = "\n###END###\n"


def load_templates(k):
    shots_length = list(range(len(SUBSEQ_EVENT_TEMPLATE.split(EXAMPLE_SPLITTER)[:-1])))
    assert len(shots_length) >= k
    indecies = random.sample(shots_length, k=k)

    prompts = {
        "subseq_event": SUBSEQ_EVENT_TEMPLATE.split(EXAMPLE_SPLITTER)[:-1],
        "subseq_event_clipped": SUBSEQ_EVENT_TEMPLATE.split(EXAMPLE_SPLITTER)[:-1],
        "cause": CAUSE_TEMPLATE.split(EXAMPLE_SPLITTER)[:-1],
        "prerequisite": PREREQUISITE_TEMPLATE.split(EXAMPLE_SPLITTER)[:-1],
        "reaction": REACTION_TEMPLATE.split(EXAMPLE_SPLITTER)[:-1],
        "motivation": MOTIVATION_TEMPLATE.split(EXAMPLE_SPLITTER)[:-1],
        }
    
    for k, v in prompts.items():
        prompt = "\n\n\n".join([v[i] for i in indecies])
        prompt += "\n\n\nContext:\n[CONTEXT]\n\nTarget:\n[TARGET]\nQuestion:\n[QUESTION]\nAnswer:\n"
        prompts[k] = prompt
    
    return prompts, indecies


def compute_metrics(scorers, predictions, golds):
    refs, hyps = {}, {}
    task_scores = {}
    for j in range(len(golds)):
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


def main(args):
    # load model
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda:0")
    model.to(dtype=torch.float16, device=device)
    # load datasets
    data = load_dataset('src/data_utils/cicero.py', 'cicero_nlg')['test']
    # load prompts
    templates, selected_idx = load_templates(args.k)

    stop_token = tokenizer.eos_token

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    core_filename = f"{args.save_path}/test"
    golds_filename = core_filename + "_gold.txt"
    generations_filename = core_filename + "_generation.txt"
    scores_filename = core_filename + "_scores.json"

    generated_sequences, gold_sequences = [], []
    for idx, row in enumerate(tqdm(data)):
        prefix = templates[row["relation"]]
        dialogue = row["dialogue"].replace(" <", "\n<")
        prompt = prefix.replace("[CONTEXT]", dialogue).replace("[TARGET]", row["target"]).replace("[QUESTION]", row["question"])
        # tokenize
        input_ids = tokenizer(prompt, return_tensors='pt')
        input_gen_len = input_ids["input_ids"].shape[1]
        # generate
        generation = model.generate(
            input_ids=input_ids['input_ids'].to(device),
            attention_mask=input_ids['attention_mask'].to(device),
            pad_token_id=tokenizer.eos_token_id,
            max_length=input_gen_len+args.max_gen_len)
        text = tokenizer.decode(generation[0, input_gen_len:], skip_special_tokens=True) 
        text = text[: text.find(stop_token) if stop_token and text.find(stop_token)>0 else None]
        text = text[: text.find('\n')]
        generated_sequences.append(text)
        gold_sequences.append(row["answer"])
        if args.debug:
            print(f"The generated sentence is: {text}")
            print("The golden sentence is:", row["answer"])
            print("="*80)
            input()
            
    with open(generations_filename, "w") as f:
        for line in generated_sequences:
            f.write(line.replace("\n", " ")+"\n")
    with open(golds_filename, "w") as f:
        for line in gold_sequences:
            f.write(line.replace("\n", " ")+"\n")

    # Let's compute scores!
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    scores = compute_metrics(scorers, generated_sequences, gold_sequences)
    scores['sample_size'] = len(data)

    keys, values = [], []
    for k,v in scores.items():
        keys.append(k)
        values.append(str(round(v*100,2)))
    scores["selected_idx"] = selected_idx
    print(" ".join(keys))
    print(" ".join(values))
        
    with open(scores_filename, "w") as f:
        json.dump(scores, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enter DEBUG mode")
    parser.add_argument("--save_path", type=str, help="where to save generations", required=True)
    parser.add_argument("--model_name_or_path", type=str, default="EleutherAI/gpt-j-6B", required=False)
    parser.add_argument("--max_gen_len", type=int, default=50, required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--k", type=int, default=3, required=False, help="k-shots for few-shot")
    args = parser.parse_args()

    set_seed(args.seed)

    main(args)
